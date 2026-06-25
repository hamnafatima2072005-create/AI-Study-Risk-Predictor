
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = int(request.form["attendance"])
    sleep_hours = float(request.form["sleep_hours"])
    assignments = int(request.form["assignments"])
    quiz_marks = int(request.form["quiz_marks"])
    screen_time = int(request.form["screen_time"])
    participation = int(request.form["participation"])

    input_data = pd.DataFrame([[
        study_hours,
        attendance,
        sleep_hours,
        assignments,
        quiz_marks,
        screen_time,
        participation
    ]], columns=[
        "study_hours",
        "attendance",
        "sleep_hours",
        "assignments",
        "quiz_marks",
        "screen_time",
        "participation"
    ])

    prediction = model.predict(input_data)[0]

    # Risk score
    risk_score = 0
    if study_hours < 4:
        risk_score += 20
    if attendance < 75:
        risk_score += 20
    if sleep_hours < 6:
        risk_score += 15
    if assignments < 60:
        risk_score += 15
    if quiz_marks < 12:
        risk_score += 15
    if screen_time > 6:
        risk_score += 15

    risk_score = min(risk_score, 100)

    advice = []
    motivation = ""

    if prediction == "High":
        advice = [
            "Increase study hours",
            "Improve attendance",
            "Reduce screen time",
            "Sleep properly"
        ]
        motivation = "Small steps daily can change everything 💪"

    elif prediction == "Medium":
        advice = [
            "Revise weak topics",
            "Improve consistency",
            "Reduce distractions"
        ]
        motivation = "You are closer to success than you think 🌱"

    else:
        advice = [
            "Maintain routine",
            "Keep practicing",
            "Aim higher"
        ]
        motivation = "Discipline today, success tomorrow 🚀"

    return render_template(
        "index.html",
        prediction=prediction,
        risk_score=risk_score,
        advice=advice,
        motivation=motivation
    )


if __name__ == "__main__":
    app.run(debug=True)