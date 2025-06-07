# app/reward_engine.py

reward_rules = {
    "streak": ["login_streak >= 3", "quizzes_completed >= 1"],
    "engagement": ["upvotes_received >= 10"],
    "social": ["karma_spent >= 25"],
    "comeback": ["login_streak == 1", "posts_created >= 2"],
    "mystery": ["karma_earned >= 15", "posts_created >= 2"],
    "allrounder": ["login_streak >= 3", "posts_created >= 5", "quizzes_completed >= 2", "upvotes_received >= 8"]
}

def determine_box_type(f):
    box=""
    if f["login_streak"]==1:
        box="comeback"
        return box
    elif f["login_streak"]>=3 and f["posts_created"]>=5 and f["quizzes_completed"]>=2 and f["upvotes_received"]>= 8:
        box="allrounder"
        return box
    elif f["karma_earned"]>=15 and (f["posts_created"]>=2 or f["quizzes_completed"]>=2 or f["login_streak"]>5):
        box="mystery"
        return box
    elif f["karma_spent"]>=25:
        box="social"
        return box
    elif f["login_streak"]>=3 and f["quizzes_completed"]>=1:
        box="streak"
        return box
    elif f["upvotes_received"]>=10:
        box="engagement"
        return box
    else:
        return ""

def generate_reason(karma,f, box):
    if not box:
        return "no_reason"
    t = []
    if box=="comeback":
        if karma>=20:
            t.append("welcome login + high activity")
        else:
            t.append("welcome login + activity")
    elif box=="allrounder":
        if f["login_streak"] >= 7 and f["posts_created"] >= 7 and f["quizzes_completed"] >= 5 and f["upvotes_received"] >= 12:
            t.append("high streak + more posts created & quizes completed + high upvotes")
        else:
            t.append("login streak +posts creation + quizes + upvotes")
    elif box=="engagement":
        if f["upvotes_received"] >= 20:
            t.append("high upvotes")
        elif f["upvotes_received"]>=15:
            t.append("moderate upvotes")
        else:
            t.append("upvotes")
    elif box=="social":
        if f["karma_spent"]>=35:
            t.append("more karma spent")
        elif f["karma_spent"]>=30:
            t.append("moderate karma spent")
        else:
            t.append("karma spent")
    elif box=="streak":
        if f["login_streak"] >= 8 and karma >= 20:
            t.append("engagement streak + high activity")
        elif f["login_streak"] >= 6:
            t.append("engagement streak")
        else:
            t.append("good login streak")
    elif box=="mystery":
        if karma>=15 and f["login_streak"]>=5:
            t.append("engagement streak + high activity")
        elif karma>=15 and f["posts_created"]>=2:
            t.append("high activity + posts created")
        elif karma>=15 and f["quizzes_completed"]>=2:
            t.append("high activity + more quizzes completed")
        else:
            t.append("high activity")
    else:
        t.append("no_reason")

    return t[0] 
def determine_rarity(karma, box, f):
    if box == "allrounder":
        if f["login_streak"] >= 7 and f["posts_created"] >= 7 and f["quizzes_completed"] >= 5 and f["upvotes_received"] >= 12:
            return "legendary"
        return "rare"
    if box == "engagement":
        if f["upvotes_received"] >= 20:
            return "legendary"
        elif f["upvotes_received"] >= 15:
            return "rare"
        return "common"
    if box == "social":
        if f["karma_spent"] >= 35:
            return "legendary"
        elif f["karma_spent"] >= 30:
            return "rare"
        return "common"
    if box == "streak":
        if f["login_streak"] >= 8 and karma >= 20:
            return "legendary"
        elif f["login_streak"] >= 6:
            return "rare"
        return "common"
    if box == "comeback":
        if karma >= 20 :
            return "rare"
        else:
            return "common"
    if box == "mystery":
        if karma >=15:
            return "rare"
        else:
            return "common"
    return "common"

def generate_status(unlocked):
    return "delivered" if unlocked else "missed"
