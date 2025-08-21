import pandas as pd
from collections import defaultdict
import random

def generate_schedule(hours_per_subject, preferences, sleep_hours_per_day=8.0, physical_activity_hours_per_day=1.0):
    # Normalize subject names to lowercase in hours_per_subject
    hours_per_subject = {k.lower(): v for k, v in hours_per_subject.items()}
    
    # Round sleep and physical activity hours
    sleep_hours_per_day = int(sleep_hours_per_day) + (1 if sleep_hours_per_day % 1 > 0.5 else 0)
    physical_activity_hours_per_day = int(physical_activity_hours_per_day) + (1 if physical_activity_hours_per_day % 1 > 0.5 else 0)

    # Define time slots: 7 days × 24 hours
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(24))
    time_slots = [(day, hour) for day in days for hour in hours]

    # School hours: Mon-Sat, 9:00-14:00
    school_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    school_hours = list(range(9, 15))
    school_slots = {(day, hour): "School" for day in school_days for hour in school_hours}

    # Calculate total hours
    total_slots = len(time_slots)  # 168
    total_school_slots = len(school_slots)  # 36
    total_study_hours = sum(hours_per_subject.values())
    total_sleep_hours = sleep_hours_per_day * 7
    total_physical_activity_hours = physical_activity_hours_per_day * 7

    # Validate total hours
    total_reserved_hours = total_school_slots + total_sleep_hours + total_physical_activity_hours + total_study_hours
    if total_reserved_hours > total_slots:
        print(f"Error: Total hours ({total_reserved_hours}) exceed available slots ({total_slots}).")
        return None

    if total_study_hours > total_slots - (total_reserved_hours - total_study_hours):
        print(f"Error: Total study hours ({total_study_hours}) exceed available slots ({total_slots - (total_reserved_hours - total_study_hours)}).")
        return None
    
    print(f"Total hours: {total_reserved_hours}, Total slots: {total_slots}, Total study hours: {total_study_hours}")

    hourdetails = {"total_slots": total_slots, "total_school_slots": total_school_slots, "total_study_hours": total_study_hours, "total_sleep_hours": total_sleep_hours, "total_physical_activity_hours": total_physical_activity_hours}

    # Initialize schedule and remaining slots
    schedule = school_slots.copy()
    remaining_slots = [slot for slot in time_slots if slot not in school_slots]

    # Calculate available slots per day
    slots_per_day = defaultdict(list)
    for slot in remaining_slots:
        slots_per_day[slot[0]].append(slot)
    available_slots_per_day = {day: len(slots) for day, slots in slots_per_day.items()}

    # Allocate sleep (prefer 22:00-6:00)
    sleep_slots_per_day = {}
    for day in days:
        sleep_slots = []
        hours_needed = sleep_hours_per_day
        for hour in list(range(22, 24)) + list(range(0, 6)) + list(range(6, 22)):
            slot = (day, hour)
            if slot in remaining_slots and len(sleep_slots) < hours_needed:
                sleep_slots.append(slot)
        if len(sleep_slots) < hours_needed:
            print(f"Warning: Only {len(sleep_slots)} sleep slots allocated on {day} (needed {hours_needed}).")
        sleep_slots_per_day[day] = sleep_slots
        for slot in sleep_slots:
            schedule[slot] = "Sleep"
            remaining_slots.remove(slot)

    # Allocate physical activity (prefer 18:00-20:00)
    activity_slots_per_day = {}
    for day in days:
        activity_slots = []
        hours_needed = physical_activity_hours_per_day
        for hour in list(range(18, 21)) + list(range(0, 18)) + list(range(21, 24)):
            slot = (day, hour)
            if slot in remaining_slots and len(activity_slots) < hours_needed:
                activity_slots.append(slot)
        if len(activity_slots) < hours_needed:
            print(f"Warning: Only {len(activity_slots)} activity slots allocated on {day} (needed {hours_needed}).")
        activity_slots_per_day[day] = activity_slots
        for slot in activity_slots:
            schedule[slot] = "Physical_Activity"
            remaining_slots.remove(slot)

    # Pre-process preferences (normalize subject to lowercase)
    allowed_slots_per_subject = {subject.lower(): remaining_slots.copy() for subject in hours_per_subject}
    restricted_subjects = set()
    for pref in preferences:
        subject = pref["subject"].lower()
        if subject not in hours_per_subject:
            continue
        constraint_type = pref["type"]
        day = pref.get("day")
        constraint_days = [day] if day else days

        if constraint_type == "exclude_subject_day":
            allowed_slots_per_subject[subject] = [
                slot for slot in allowed_slots_per_subject[subject] if slot[0] != day
            ]
        elif constraint_type == "restrict_subject_to_hours":
            allowed_hours = pref.get("allowed_hours", [])
            allowed_slots_per_subject[subject] = [
                slot for slot in allowed_slots_per_subject[subject]
                if slot[0] in constraint_days and slot[1] in allowed_hours
            ]
            restricted_subjects.add(subject)
        elif constraint_type == "exclude_subject_time":
            hour = pref.get("hour")
            allowed_slots_per_subject[subject] = [
                slot for slot in allowed_slots_per_subject[subject]
                if not (slot[0] == day and slot[1] == hour)
            ]

    # Validate and handle restrictive preferences
    for subject, slots in allowed_slots_per_subject.items():
        hours_needed = hours_per_subject[subject]
        if len(slots) < hours_needed:
            if subject in restricted_subjects:
                print(f"Warning: Only {len(slots)} slots available for {subject} (needed {hours_needed}). Using fallback slots.")
                allowed_slots_per_subject[subject] = remaining_slots.copy()
                for pref in preferences:
                    if pref["subject"].lower() != subject:
                        continue
                    if pref["type"] == "exclude_subject_day":
                        allowed_slots_per_subject[subject] = [
                            slot for slot in allowed_slots_per_subject[subject] if slot[0] != pref["day"]
                        ]
                    elif pref["type"] == "exclude_subject_time":
                        allowed_slots_per_subject[subject] = [
                            slot for slot in allowed_slots_per_subject[subject]
                            if not (slot[0] == pref["day"] and slot[1] == pref["hour"])
                        ]
            if len(allowed_slots_per_subject[subject]) < hours_needed:
                print(f"Error: Still only {len(allowed_slots_per_subject[subject])} slots available for {subject} (needed {hours_needed}).")
                return None

    # Resolve conflicts with sleep/activity
    for subject, slots in allowed_slots_per_subject.items():
        hours_needed = hours_per_subject[subject]
        conflicting_slots = [slot for slot in slots if slot in schedule]
        if conflicting_slots:
            print(f"Adjusting sleep/activity for {subject} due to {len(conflicting_slots)} conflicting slots.")
            for slot in conflicting_slots:
                current_activity = schedule[slot]
                conflicting_day = slot[0]
                replacement_slot = next(
                    (s for s in slots_per_day[conflicting_day] if s not in schedule and s in remaining_slots),
                    None
                )
                if replacement_slot:
                    schedule[replacement_slot] = current_activity
                    del schedule[slot]
                    remaining_slots.remove(replacement_slot)
                    remaining_slots.append(slot)
                    if current_activity == "Sleep":
                        sleep_slots_per_day[conflicting_day].remove(slot)
                        sleep_slots_per_day[conflicting_day].append(replacement_slot)
                    else:
                        activity_slots_per_day[conflicting_day].remove(slot)
                        activity_slots_per_day[conflicting_day].append(replacement_slot)
                else:
                    for other_day in days:
                        if other_day == conflicting_day:
                            continue
                        replacement_slot = next(
                            (s for s in slots_per_day[other_day] if s not in schedule and s in remaining_slots),
                            None
                        )
                        if replacement_slot:
                            schedule[replacement_slot] = current_activity
                            del schedule[slot]
                            remaining_slots.remove(replacement_slot)
                            remaining_slots.append(slot)
                            if current_activity == "Sleep":
                                sleep_slots_per_day[conflicting_day].remove(slot)
                                sleep_slots_per_day[other_day].append(replacement_slot)
                            else:
                                activity_slots_per_day[conflicting_day].remove(slot)
                                activity_slots_per_day[other_day].append(replacement_slot)
                            break
        allowed_slots_per_subject[subject] = [slot for slot in slots if slot not in schedule]

        if len(allowed_slots_per_subject[subject]) < hours_needed:
            print(f"Error: Only {len(allowed_slots_per_subject[subject])} slots available for {subject} after conflict resolution (needed {hours_needed}).")
            return None

    # Distribute study hours evenly across days
    target_hours_per_day = {}
    for subject, hours_needed in hours_per_subject.items():
        allowed_days = set(slot[0] for slot in allowed_slots_per_subject[subject])
        if not allowed_days:
            print(f"Error: No allowed days for {subject}.")
            return None
        base_hours = hours_needed // len(allowed_days)
        extra_hours = hours_needed % len(allowed_days)
        target_hours_per_day[subject] = {day: base_hours + (1 if i < extra_hours else 0) for i, day in enumerate(allowed_days)}

    # Greedy assignment for study hours
    for subject, hours_needed in hours_per_subject.items():
        hours_assigned = 0
        allowed_days = list(set(slot[0] for slot in allowed_slots_per_subject[subject]))
        random.shuffle(allowed_days)
        hours_per_day = target_hours_per_day[subject]
        
        if subject in restricted_subjects:
            preferred_slots = [s for s in allowed_slots_per_subject[subject] if s in remaining_slots]
            for slot in preferred_slots[:hours_needed]:
                if hours_assigned < hours_needed:
                    schedule[slot] = subject
                    remaining_slots.remove(slot)
                    hours_assigned += 1
        
        for day in allowed_days:
            if hours_assigned >= hours_needed:
                break
            day_slots = [s for s in allowed_slots_per_subject[subject] if s[0] == day and s in remaining_slots]
            random.shuffle(day_slots)
            max_slots = hours_per_day.get(day, 0)
            for slot in day_slots[:max_slots]:
                if hours_assigned < hours_needed:
                    schedule[slot] = subject
                    remaining_slots.remove(slot)
                    hours_assigned += 1
        
        if hours_assigned < hours_needed:
            for slot in allowed_slots_per_subject[subject]:
                if slot in remaining_slots and hours_assigned < hours_needed:
                    schedule[slot] = subject
                    remaining_slots.remove(slot)
                    hours_assigned += 1

        if hours_assigned < hours_needed:
            print(f"Error: Could only assign {hours_assigned} of {hours_needed} hours for {subject}.")
            return None

    # Validate schedule
    actual_study_hours = sum(1 for act in schedule.values() if act in hours_per_subject)
    if actual_study_hours != total_study_hours:
        print(f"Error: Expected {total_study_hours} study hours, got {actual_study_hours}.")
        return None

    # Convert to DataFrame
    schedule_list = [
        {"Day": day, "Hour": hour, "Activity": schedule.get((day, hour), "Free")}
        for day, hour in time_slots
    ]
    df = pd.DataFrame(schedule_list)
    df.sort_values(by=["Day", "Hour"], inplace=True)
    hours_day = {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 0, "Fri": 0, "Sat": 0, "Sun": 0}
    for i in range(len(df)):
        if df["Activity"][i] != "Free" and df["Activity"][i] != "School" and df["Activity"][i] != "Sleep" and df["Activity"][i] != "Physical_Activity":
            hours_day[df["Day"][i]] += 1
    
    return df, hourdetails, hours_day

# if __name__ == "__main__":
#     hours_per_subject = {'math': 6, 'physics': 4, 'chemistry': 7, 'english': 3, 'history': 4}
#     preferences = [
#         {'type': 'exclude_subject_day', 'subject': 'Math', 'day': 'Sun'},
#         {'type': 'exclude_subject_day', 'subject': 'Math', 'day': 'Thu'},
#         {'type': 'restrict_subject_to_hours', 'subject': 'English', 'allowed_hours': [18, 19], 'day': 'Wed'},
#         {'type': 'exclude_subject_time', 'subject': 'Chemistry', 'day': 'Fri', 'hour': 17},
#         {'type': 'restrict_subject_to_hours', 'subject': 'Chemistry', 'allowed_hours': [15, 16, 17], 'day': 'Wed'},
#         {'type': 'exclude_subject_day', 'subject': 'Physics', 'day': 'Thu'}
#     ]
#     sleep_hours_per_day = 7
#     physical_activity_hours_per_day = 3

#     schedule_df, hourdetails, hours_day = generate_schedule(
#         hours_per_subject,
#         preferences,
#         sleep_hours_per_day,
#         physical_activity_hours_per_day
#     )
#     if schedule_df is not None:
#         print(schedule_df)
#         print("Hour details:", hourdetails)
#         print("Hours per day:", hours_day)