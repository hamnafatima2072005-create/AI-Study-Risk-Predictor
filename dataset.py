import pandas as pd
import random

data = []

for _ in range(1000):

    # Screen time (1–10)
    screen_time = random.randint(1, 10)

    # Study hours inversely related to screen time + noise
    study_hours = round(max(1, min(8, 8 - screen_time * 0.5 + random.uniform(-2, 2))), 1)

    # Sleep hours
    sleep_hours = round(random.uniform(4, 9), 1)

    # Attendance
    attendance = random.randint(40, 100)

    # Participation correlated with attendance but not exact
    if attendance > 80:
        participation = random.choice([1, 2, 2])
    elif attendance > 60:
        participation = random.choice([0, 1, 1])
    else:
        participation = random.choice([0, 0, 1])

    # Assignments with noise
    assignments = study_hours * 10 + attendance * 0.35 + random.uniform(-20, 20)
    assignments = int(max(0, min(100, assignments)))

    # Quiz marks with more randomness
    quiz_marks = study_hours * 1.5 + sleep_hours * 0.8 + random.uniform(-5, 5)
    quiz_marks = int(max(0, min(20, quiz_marks)))

    # -------- Risk score --------
    risk_score = 0

    risk_score += max(0, 4 - study_hours) * 0.8
    risk_score += max(0, 65 - attendance) * 0.05
    risk_score += max(0, 6 - sleep_hours) * 0.7
    risk_score += max(0, 60 - assignments) * 0.03
    risk_score += max(0, 12 - quiz_marks) * 0.4
    risk_score += max(0, screen_time - 6) * 0.5
    risk_score += (2 - participation) * 0.6

    # Random noise (important)
    risk_score += random.uniform(-1.5, 1.5)

    # Final label (soft boundaries)
    if risk_score > 5.8:
        risk_level = "High"
    elif risk_score > 3.2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    data.append([
        study_hours,
        attendance,
        sleep_hours,
        assignments,
        quiz_marks,
        screen_time,
        participation,
        risk_level
    ])

df = pd.DataFrame(data, columns=[
    "study_hours",
    "attendance",
    "sleep_hours",
    "assignments",
    "quiz_marks",
    "screen_time",
    "participation",
    "risk_level"
])

df.to_csv("student_risk_dataset.csv", index=False)

print(df.head())
print(df["risk_level"].value_counts())