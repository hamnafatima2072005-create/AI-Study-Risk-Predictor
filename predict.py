import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("student_risk_dataset.csv")

# Features and target
X = df.drop("risk_level", axis=1)
y = df["risk_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------- User Input --------
study_hours = float(input("Enter study hours: "))
attendance = int(input("Enter attendance (%): "))
sleep_hours = float(input("Enter sleep hours: "))
assignments = int(input("Enter assignments completion (%): "))
quiz_marks = int(input("Enter quiz marks (0-20): "))
screen_time = int(input("Enter screen time (hours): "))
participation = int(input("Enter participation (0=Low, 1=Medium, 2=High): "))

# Create input dataframe
user_data = pd.DataFrame([[
    study_hours,
    attendance,
    sleep_hours,
    assignments,
    quiz_marks,
    screen_time,
    participation
]], columns=X.columns)

# Predict
prediction = model.predict(user_data)

print("\nPrediction:", prediction[0])

# Friendly message
if prediction[0] == "High":
    print("\n🚨 HIGH ACADEMIC RISK")
    print("Your current habits show a high chance of poor academic performance.")
    print("\nAdvice:")
    
    if study_hours < 4:
        print("- Increase study hours to at least 4-5 hours/day.")
    
    if attendance < 75:
        print("- Improve class attendance. Missing lectures hurts learning.")
    
    if sleep_hours < 6:
        print("- Sleep at least 6-8 hours for better focus.")
    
    if assignments < 60:
        print("- Complete assignments on time.")
    
    if quiz_marks < 12:
        print("- Revise weak topics before quizzes.")
    
    if screen_time > 6:
        print("- Reduce social media / mobile usage.")

    print("\nMotivation:")
    print("Bad habits today do NOT define your future.")
    print("Small improvements every day can completely change your result.")
    print("Start with one step today. You can do this! 💪")


elif prediction[0] == "Medium":
    print("\n⚠️ MEDIUM ACADEMIC RISK")
    print("You are doing okay, but there is room for improvement.")

    print("\nAdvice:")
    
    if study_hours < 4:
        print("- Try studying 1-2 more hours daily.")
    
    if attendance < 80:
        print("- Attend more classes regularly.")
    
    if screen_time > 5:
        print("- Reduce screen time for better focus.")
    
    if assignments < 80:
        print("- Improve assignment completion.")

    print("\nMotivation:")
    print("You are not far from becoming a top-performing student.")
    print("Consistency matters more than perfection. Keep improving! 🌱")


else:
    print("\n✅ LOW ACADEMIC RISK")
    print("Great job! Your academic habits look strong.")

    print("\nAdvice:")
    print("- Maintain your current routine.")
    print("- Keep revising regularly.")
    print("- Help classmates and strengthen concepts.")

    print("\nMotivation:")
    print("Success comes from discipline repeated every day.")
    print("Keep growing and aim even higher! 🚀")