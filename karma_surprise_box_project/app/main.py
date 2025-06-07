# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.model_utils import predict_surprise
from app.reward_engine import determine_box_type, generate_reason, determine_rarity, generate_status

app = FastAPI()

class DailyMetrics(BaseModel):
    login_streak: int
    posts_created: int
    comments_written: int
    upvotes_received: int
    quizzes_completed: int
    buddies_messaged: int
    karma_spent: int
    karma_earned: int
    spam: bool

class SurpriseRequest(BaseModel):
    user_id: str
    date: str
    daily_metrics: DailyMetrics

@app.post("/check-surprise-box")
def check_surprise_box(req: SurpriseRequest):
    f = req.daily_metrics.dict()
    surprise_unlocked = predict_surprise(f)

    if surprise_unlocked:
        box_type = determine_box_type(f)
        karma = abs(f["karma_earned"] - f["karma_spent"])
        reason = generate_reason(karma,f, box_type)
        rarity = determine_rarity(karma, box_type, f)
    else:
        box_type = ""
        reason = "spam" if f["spam"] else "low_karma_diff"
        karma = 0
        rarity = ""

    return {
        "user_id": req.user_id,
        "surprise_unlocked": surprise_unlocked,
        "reward_karma": karma,
        "reason": reason,
        "rarity": rarity,
        "box_type": box_type,
        "status": generate_status(surprise_unlocked)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    with open("app/version.txt") as f:
        return {"version": f.read().strip()}
