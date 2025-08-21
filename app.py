from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response, jsonify
import json
import os
import pandas as pd
from csp_scheduler import generate_schedule
from ml_classifier import train_classifier, train_score_predictor, predict_weak_subject, predict_future_score_and_health, recommend_study_hours, predpassorfail
from schedule_visualizer import visualize_schedule
from datetime import datetime
from emotion_detector import generate_frames, get_latest_recommendation

app = Flask(__name__)
app.secret_key = "your-secret-key"
USER_FILE = "users.json"
SCORE_DIR = "scores"  # Directory to store individual score files

# Ensure scores directory exists
if not os.path.exists(SCORE_DIR):
    os.makedirs(SCORE_DIR)

# Initialize users file
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        gender = request.form.get("gender").strip()
        action = request.form.get("action")

        with open(USER_FILE, "r") as f:
            users = json.load(f)
            
        if username == "" or password == "":
            return render_template("login.html", error="Please fill in all fields")
        
        elif action == "login":
            if username in users and users[username]["password"] == password:
                session["username"] = username
                return redirect(url_for("planner"))
            else:
                return render_template("login.html", error="Invalid username or password")
        elif action == "signup":
            if username in users:
                return render_template("login.html", error="Username already exists")
            users[username] = {
                "password": password,
                "Gender": gender,
                "subjects": ["Math", "Physics", "Chemistry", "English", "History"],
                "constraints": [],
                "hours": {subj.lower().replace(" ", "_"): 0 for subj in ["Math", "Physics", "Chemistry", "English", "History"]},
                "scores": {subj.lower().replace(" ", "_"): 80 for subj in ["Math", "Physics", "Chemistry", "English", "History"]},
                "preferences": [],
                "study_time": 0,
                "attendance": 90,
                "tutoring_sessions": 2,
                "sleep_hours": 7.0,  # Default value
                "physical_activity": 3.0,  # Default value
                "school_type": "Public",
                "parental_education_level": "",
                "teacher_quality": "",
                "prev_exam_score": 80.0
            }
            with open(USER_FILE, "w") as f:
                json.dump(users, f, indent=4)
            return render_template("login.html", success="Sign-up successful! Please log in.")
        else:
            return render_template("login.html", error="Please fill in all fields")

    return render_template("login.html")

@app.route("/planner", methods=["GET", "POST"])
def planner():
    if "username" not in session:
        return redirect(url_for("login"))

    # Load user-specific data
    with open(USER_FILE, "r") as f:
        users = json.load(f)
    username = session["username"]
    user_data = users[username]
    subjects = user_data["subjects"]
    constraints = user_data["constraints"]

    # Load existing score history for the user from CSV
    score_file = os.path.join(SCORE_DIR, f"f.{username}.csv")
    score_history = []
    if os.path.exists(score_file):
        df = pd.read_csv(score_file)
        score_history = df.to_dict('records')  # Convert to list of dictionaries

    # Train ML models
    clf = train_classifier("data/student_scores.csv", username=username)
    score_predictor = train_score_predictor("data/student_performance.csv", username=username)

    # Initialize form data from user_data (stored in users.json)
    form_data = {
        "hours": user_data.get("hours", {subj.lower().replace(" ", "_"): 0 for subj in subjects}),
        "scores": user_data.get("scores", {subj.lower().replace(" ", "_"): 80 for subj in subjects}),
        "score_history": score_history,  # Load from CSV
        "preferences": user_data.get("preferences", []),
        "study_time": user_data.get("study_time", 0),
        "attendance": user_data.get("attendance", 90),
        "tutoring_sessions": user_data.get("tutoring_sessions", 2),
        "sleep_hours": user_data.get("sleep_hours", 7.0),
        "physical_activity": user_data.get("physical_activity", 3.0),
        "school_type": user_data.get("school_type", "Public"),
        "parental_education_level": user_data.get("parental_education_level", ""),
        "teacher_quality": user_data.get("teacher_quality", ""),
        "prev_exam_score": user_data.get("prev_exam_score", 80.0),
        "gender": user_data.get("Gender", "Male")
    }

    if request.method == "POST":
        action = request.form.get("action")

        # Process form data
        hours = {}
        scores = {}
        preferences = []
        for subj in subjects:
            subj_key = subj.lower().replace(" ", "_")
            hours[subj_key] = int(request.form.get(f"hours_{subj_key}", 0))
            scores[subj_key] = int(request.form.get(f"score_{subj_key}", 0))
        for constraint in constraints:
            constraint_id = f"constraint_{constraint['type']}" if "subject" not in constraint else f"constraint_{constraint['type']}_{constraint['subject']}"
            if request.form.get(constraint_id):
                preferences.append(constraint_id)
        
        study_time = int(request.form.get("study_time", 0))
        attendance = int(request.form.get("attendance", 90))
        tutoring_sessions = int(request.form.get("tutoring_sessions", 2))
        sleep_hours = float(request.form.get("sleep_hours", 7.0))
        physical_activity = float(request.form.get("physical_activity", 3.0))
        school_type = request.form.get("school_type", "Public")
        parental_education_level = request.form.get("parental_education_level", "")
        teacher_quality = request.form.get("teacher_quality", "")

        # Calculate average score
        average_score = sum(scores.values()) / len(scores) if scores else 0

        # Update form_data with submitted values
        form_data = {
            "hours": hours,
            "scores": scores,
            "score_history": score_history,  # Load from CSV
            "preferences": preferences,
            "study_time": study_time,
            "attendance": attendance,
            "tutoring_sessions": tutoring_sessions,
            "sleep_hours": sleep_hours,
            "physical_activity": physical_activity,
            "school_type": school_type,
            "parental_education_level": parental_education_level,
            "teacher_quality": teacher_quality,
            "prev_exam_score": round(average_score, 2),
            "gender": user_data["Gender"]
        }

        # Update user_data and save to users.json
        user_data["hours"] = hours
        user_data["scores"] = scores
        user_data["preferences"] = preferences
        user_data["study_time"] = study_time
        user_data["attendance"] = attendance
        user_data["tutoring_sessions"] = tutoring_sessions
        user_data["sleep_hours"] = sleep_hours
        user_data["physical_activity"] = physical_activity
        user_data["school_type"] = school_type
        user_data["parental_education_level"] = parental_education_level
        user_data["teacher_quality"] = teacher_quality
        user_data["prev_exam_score"] = round(average_score, 2)
        users[username] = user_data
        with open(USER_FILE, "w") as f:
            json.dump(users, f, indent=4)

        # Add new subject
        if action == "add_subject":
            new_subject = request.form.get("new_subject").strip()
            if new_subject and new_subject not in subjects:
                subjects.append(new_subject)
                user_data["subjects"] = subjects
                user_data["hours"][new_subject.lower().replace(" ", "_")] = 0
                user_data["scores"][new_subject.lower().replace(" ", "_")] = 80
                # Recalculate average score after adding new subject
                scores[new_subject.lower().replace(" ", "_")] = 80
                average_score = sum(scores.values()) / len(scores) if scores else 0
                user_data["prev_exam_score"] = round(average_score, 2)
                users[username] = user_data
                with open(USER_FILE, "w") as f:
                    json.dump(users, f, indent=4)
            return redirect(url_for("planner"))

        # Add new constraint
        elif action == "add_constraint":
            constraint_type = request.form.get("constraint_type").strip()
            subject = request.form.get("constraint_subject").strip()
            if constraint_type and subject in subjects:
                new_constraint = {"type": constraint_type, "subject": subject}
                if constraint_type == "exclude_subject_time":
                    hour = request.form.get("constraint_hour")
                    day = request.form.get("constraint_day")
                    if hour and day:
                        new_constraint["day"] = day
                        new_constraint["hour"] = int(hour)
                    else:
                        return render_template("planner.html", error="Hour and day are required for exclude_subject_time.", subjects=subjects, constraints=constraints, form_data=form_data)
                elif constraint_type == "restrict_subject_to_hours":
                    allowed_hours = request.form.getlist("constraint_allowed_hours")
                    day = request.form.get("constraint_day")
                    if allowed_hours:
                        new_constraint["allowed_hours"] = [int(h) for h in allowed_hours]
                        # Remove duplicates and sort
                        new_constraint["allowed_hours"] = sorted(set(new_constraint["allowed_hours"]))
                        new_constraint["day"] = day
                    else:
                        return render_template("planner.html", error="Allowed hours are required for restrict_subject_to_hours.", subjects=subjects, constraints=constraints, form_data=form_data)
                elif constraint_type == "exclude_subject_day":
                    day = request.form.get("constraint_day")
                    if day:
                        new_constraint["day"] = day
                    else:
                        return render_template("planner.html", error="Day is required for exclude_subject_day.", subjects=subjects, constraints=constraints, form_data=form_data)
                constraints.append(new_constraint)
                user_data["constraints"] = constraints
                users[username] = user_data
                with open(USER_FILE, "w") as f:
                    json.dump(users, f, indent=4)
            return redirect(url_for("planner"))

        # Generate schedule
        elif action == "generate":
            try:
                # Get scores for ML prediction
                scores_for_ml = {}
                for subj in subjects:
                    score_value = request.form.get(f"score_{subj.lower().replace(' ', '_')}")
                    if score_value is None:
                        raise ValueError(f"Missing score for {subj}")
                    score = int(score_value)
                    if not 0 <= score <= 100:
                        raise ValueError("Scores must be between 0 and 100")
                    scores_for_ml[f"{subj}_Score"] = score

                # Predict weak subject
                weak_subject = predict_weak_subject(clf, scores_for_ml, username=username)

                # Use the stored prev_exam_score for ML prediction
                previous_exam_score = user_data["prev_exam_score"]

                # Predict health factors and scores
                features = {
                    "Study_Time": study_time,
                    "Attendance": attendance,
                    "Gender": user_data["Gender"],
                    "Previous_Exam_Score": previous_exam_score,
                    "Tutoring_Sessions": tutoring_sessions,
                    "Teacher_Quality": teacher_quality,
                    "School_Type": school_type,
                    "Parental_Education_Level": parental_education_level,
                    "Sleep_Hours": sleep_hours,
                    "Physical_Activity": physical_activity
                }
                predictions = predict_future_score_and_health(score_predictor, features, username=username)

                # Extract predictions with rounding
                exam_score = round(predictions["Future_Score"], 2)
                health_plan = {
                    "Sleep_Hours": sleep_hours,
                    "Extracurricular_Activities": predictions["Extracurricular_Activities"],
                    "Physical_Activity": physical_activity
                }

                # Generate schedule using user-provided sleep_hours and physical_activity
                print(f"Generating schedule with sleep_hours: {sleep_hours}, physical_activity: {physical_activity}")
                print(constraints)
                schedule_df, totalshours, hoursperday = generate_schedule(
                    hours_per_subject={subj.lower().replace(" ", "_"): hours[subj.lower().replace(" ", "_")] for subj in subjects},
                    preferences=constraints,
                    sleep_hours_per_day=sleep_hours,
                    physical_activity_hours_per_day=physical_activity
                )
                if schedule_df is None:
                    raise ValueError("Your total hours must be less than available hours")
                recommendations, plot_files = recommend_study_hours(clf, scores_for_ml, username, totalshours)
                
                # Save current scores to CSV
                current_scores = scores.copy()
                current_scores["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_df = pd.DataFrame([current_scores])
                if os.path.exists(score_file):
                    existing_df = pd.read_csv(score_file)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    updated_df = new_df
                updated_df.to_csv(score_file, index=False)

                # Calculate average hours per day
                total_hours = sum(hoursperday.values())
                num_days = len(hoursperday)
                average_hours = total_hours / num_days if num_days > 0 else 0
                print(f"Average hours per day: {average_hours}")
                print(f"Previous exam score: {previous_exam_score}")
                check = predpassorfail(average_hours, previous_exam_score)
                session["check"] = check
                print(f"Pass or Fail: {check}")

                # Store recommendations, predictions, schedule, and plot files in session
                session["recommendations"] = recommendations
                session["exam_score"] = exam_score
                session["health_plan"] = health_plan
                session["weak_subject"] = weak_subject
                session["schedule_data"] = schedule_df.to_dict() if schedule_df is not None else []
                session["plot_files"] = plot_files

                return redirect(url_for("schedule"))

            except ValueError as e:
                return render_template("planner.html", error=str(e), subjects=subjects, constraints=constraints, form_data=form_data)
            except Exception as e:
                return render_template("planner.html", error=f"An unexpected error occurred: {str(e)}", subjects=subjects, constraints=constraints, form_data=form_data)

    return render_template("planner.html", subjects=subjects, constraints=constraints, form_data=form_data)

@app.route("/schedule")
def schedule():
    if "username" not in session:
        return redirect(url_for("login"))

    recommendations = session.get("recommendations", {})
    schedule_data = session.get("schedule_data", [])
    exam_score = session.get("exam_score", 0)
    health_plan = session.get("health_plan", {})
    weak_subject = session.get("weak_subject", "")
    plot_files = session.get("plot_files", [])
    check = session.get("check")

    # Convert schedule_data back to DataFrame for rendering if it exists
    schedule_summary = pd.DataFrame(schedule_data).to_dict('records') if schedule_data else []

    print(f"Exam score: {exam_score}")
    print(f"Health plan: {health_plan.get('Extracurricular_Activities', 'No')}")

    return render_template("schedule.html", recommendations=recommendations, schedule_summary=schedule_summary, exam_score=exam_score, health_plan=health_plan, weak_subject=weak_subject, plot_files=plot_files, check=check)

@app.route("/face")
def face():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("face.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/recommendation")
def get_recommendation():
    return jsonify({'recommendation': get_latest_recommendation()})

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("recommendations", None)
    session.pop("schedule_data", None)
    session.pop("exam_score", None)
    session.pop("health_plan", None)
    session.pop("weak_subject", None)
    session.pop("plot_files", None)
    session.pop("check", None)
    return redirect(url_for("login"))

@app.route("/schedule.png")
def serve_schedule():
    return send_file("schedule.png")

@app.route("/plots/<username>/<filename>")
def serve_plot(username, filename):
    plot_path = os.path.join("plots", username, filename)
    return send_file(plot_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")