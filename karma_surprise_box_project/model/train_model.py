# model/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("data/karma_surprise_box_dataset.csv")

features = [
    "login_streak", "posts_created", "comments_written", "upvotes_received",
    "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned", "spam"
]
X = df[features]
y = df["surprise_unlocked"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("Evaluation Report:")
print(classification_report(y_test, y_pred))
joblib.dump(model, "model/classifier.pkl")
