import matplotlib.pyplot as plt
import pandas as pd

def visualize_schedule(schedule_df, output_path="schedule.png"):
    # Define colors for subjects
    colors = {
        "Math": "blue",
        "Physics": "red",
        "Chemistry": "green",
        "English": "purple",
        "History": "orange"
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Map days and hours to indices
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(9, 21))
    day_map = {day: i for i, day in enumerate(days)}
    hour_map = {hour: i for i, hour in enumerate(hours)}
    
    # Plot each session
    for _, row in schedule_df.iterrows():
        day_idx = day_map[row["Day"]]
        hour_idx = hour_map[row["Hour"]]
        subject = row["Subject"]
        ax.add_patch(plt.Rectangle((day_idx, hour_idx), 1, 1, color=colors[subject]))
    
    # Configure plot
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)
    ax.set_yticks(range(len(hours)))
    ax.set_yticklabels([f"{h}:00" for h in hours])
    ax.set_xlabel("Day")
    ax.set_ylabel("Hour")
    ax.set_title("Weekly Study Schedule")
    ax.invert_yaxis()  # Earliest hour at top
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[subj], label=subj) for subj in colors]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Sample schedule for testing
    sample_schedule = pd.DataFrame([
        {"Day": "Mon", "Hour": 9, "Subject": "Math"},
        {"Day": "Mon", "Hour": 10, "Subject": "Physics"},
        {"Day": "Tue", "Hour": 11, "Subject": "Chemistry"}
    ])
    visualize_schedule(sample_schedule)