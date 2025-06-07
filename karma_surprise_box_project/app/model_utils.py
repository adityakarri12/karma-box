import joblib
import os
from typing import Dict, Union

# Construct the model path
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "classifier.pkl")

# Load the model once when this module is imported
model = joblib.load(model_path)

def predict_surprise(features: Dict[str, Union[int, float]]) -> bool:
    feature_order = [
        "login_streak", "posts_created", "comments_written", "upvotes_received",
        "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned", "spam"
    ]
    try:
        X = [[features[feat] for feat in feature_order]]
    except KeyError as e:
        raise ValueError(f"Missing feature key: {e.args[0]}")
    return bool(model.predict(X)[0])
