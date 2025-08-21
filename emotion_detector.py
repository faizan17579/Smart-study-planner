import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained emotion recognition model
model = load_model('emotion_recognition_model.h5')

# Emotion labels mapping (FER2013)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Global variable to store the latest recommendation
latest_recommendation = "Ensure your face is visible to start."

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    global latest_recommendation

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            predictions = model.predict(face)
            dominant_emotion_idx = np.argmax(predictions[0])
            dominant_emotion = emotion_labels[dominant_emotion_idx]
            confidence = predictions[0][dominant_emotion_idx]

            # Update the recommendation based on the latest emotion
            if dominant_emotion == 'happy' and confidence > 0.3:
                latest_recommendation = "You seem happy! This is a great time to study and make progress."
            elif dominant_emotion == 'sad' and confidence > 0.3:
                latest_recommendation = "You seem sad. How about taking a break to relax? Maybe listen to some music."
            elif dominant_emotion in ['angry', 'fear'] and confidence > 0.3:
                latest_recommendation = "You seem stressed. Try some physical activity or a short relaxation session."
            elif dominant_emotion == 'neutral' and confidence > 0.3:
                current_hour = int(time.strftime("%H"))
                if current_hour > 21 or current_hour < 6:
                    latest_recommendation = "It’s late or early! Consider getting some sleep."
                else:
                    latest_recommendation = "You seem calm. This could be a good time to study or relax, depending on your needs."
            else:
                latest_recommendation = "Emotion detection confidence is low. Consider taking a break or studying based on your schedule."

            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            latest_recommendation = "Ensure your face is visible to start."

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def get_latest_recommendation():
    return latest_recommendation

