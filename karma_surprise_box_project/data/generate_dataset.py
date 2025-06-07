import random
import pandas as pd

# Feature names
features_list = [
    "login_streak", "posts_created", "comments_written", "upvotes_received",
    "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned", "spam"
]

# Reward rules (without 'quiz_master' and 'post_master')
reward_rules = {
    "streak": ["login_streak >= 3", "quizzes_completed >= 1"],
    "engagement": ["upvotes_received >= 10"],
    "social": ["karma_spent >= 25"],
    "comeback": ["login_streak == 1", "posts_created >= 1"],
    "mystery": ["karma_earned >= 15", "posts_created >= 1"],
    "allrounder": ["login_streak >= 3", "posts_created >= 5", "quizzes_completed >= 2", "upvotes_received >= 8"]
}

MIN_KARMA = 5
MAX_KARMA = 80

# Probability thresholds for dataset composition
SPAM_RATIO = 0.03
LOW_KARMA_DIFF_RATIO = 0.10
REWARD_TRUE_RATIO = 1 - SPAM_RATIO - LOW_KARMA_DIFF_RATIO

# Utility functions
def check_rule(features, rule):
    for var in features:
        rule = rule.replace(var, str(features[var]))
    return eval(rule)

def determine_rarity(karma, box, features):
    if box is None:
        return ""

    if box == "allrounder":
        if (features["login_streak"] >= 6 and features["posts_created"] >= 7 and
            features["quizzes_completed"] >= 5 and features["upvotes_received"] >= 12 ):
            return "legendary"
        else:
            return "rare"

    if box == "engagement":
        if features["upvotes_received"] >= 20:
            return "legendary"
        elif features["upvotes_received"] >= 15:
            return "rare"
        else:
            return "common"

    if box == "social":
        if features["karma_spent"] >= 35 :
            return "legendary"
        elif features["karma_spent"] >= 30:
            return "rare"
        else:
            return "common"

    if box == "streak":
        if features["login_streak"] >= 8 and karma >= 20:
            return "legendary"
        elif features["login_streak"] >= 6:
            return "rare"
        else:
            return "common"

    if box == "comeback":
        if karma >= 20:
            return "rare"
        else:
            return "common"

    if box == "mystery":
        if karma >= 15:
            return "rare"
        else:
            return "common"

    return "common"

def generate_reason(features, box):
    if box is None:
        return "no_reason"
    triggered_features = []
    for rule in reward_rules[box]:
        feat = rule.split()[0]
        if check_rule(features, rule):
            triggered_features.append(feat)
    triggered_features = list(set(triggered_features))
    return "+".join(triggered_features) if triggered_features else "no_reason"

def generate_status(surprise_unlocked):
    if surprise_unlocked:
        return "delivered"
    return "missed"

# Generate features with controlled randomness to trigger specific boxes and rarities
def generate_features_for_box(box=None, force_low_karma=False, force_spam=False):
    # Initialize feature dict with default values
    f = {
        "login_streak": 0,
        "posts_created": 0,
        "comments_written": 0,
        "upvotes_received": 0,
        "quizzes_completed": 0,
        "buddies_messaged": 0,
        "karma_spent": 0,
        "karma_earned": 0,
        "spam": force_spam
    }

    # If spam forced, karma values can be random but no reward unlocked
    if force_spam:
        f["karma_earned"] = random.randint(0, 80)
        f["karma_spent"] = random.randint(0, f["karma_earned"])
        # Other features can be low or random
        for feat in features_list:
            if feat not in ["karma_earned", "karma_spent", "spam"]:
                f[feat] = random.randint(0, 5)
        return f

    # For specific box types
    if box == "allrounder":
        f["login_streak"] = random.randint(3, 10)
        f["posts_created"] = random.randint(5, 15)
        f["quizzes_completed"] = random.randint(2, 10)
        f["upvotes_received"] = random.randint(8, 25)
        f["comments_written"] = random.randint(3, 20)
        f["buddies_messaged"] = random.randint(3, 15)
        f["karma_earned"] = random.randint(25, 80)
        # karma spent less than earned
        f["karma_spent"] = random.randint(0, f["karma_earned"] - MIN_KARMA)
        f["spam"] = False

    elif box == "streak":
        f["login_streak"] = random.randint(3, 10)
        f["quizzes_completed"] = random.randint(1, 5)
        f["posts_created"] = random.randint(0, 10)
        f["karma_earned"] = random.randint(15, 80)
        f["karma_spent"] = random.randint(0, f["karma_earned"] - MIN_KARMA)
        # Other features moderate
        f["comments_written"] = random.randint(0, 10)
        f["upvotes_received"] = random.randint(0, 15)
        f["buddies_messaged"] = random.randint(0, 10)
        f["spam"] = False

    elif box == "engagement":
        f["upvotes_received"] = random.randint(10, 30)
        f["comments_written"] = random.randint(5, 20)
        f["buddies_messaged"] = random.randint(3, 15)
        f["karma_earned"] = random.randint(20, 80)
        f["karma_spent"] = random.randint(0, f["karma_earned"] - MIN_KARMA)
        # Other features moderate
        f["login_streak"] = random.randint(0, 5)
        f["posts_created"] = random.randint(0, 6)
        f["quizzes_completed"] = random.randint(0, 3)
        f["spam"] = False

    elif box == "social":
        f["karma_earned"] = random.randint(25, 30)
        f["karma_spent"] = random.randint(20, f["karma_earned"] - MIN_KARMA)
        # Other features random moderate
        f["login_streak"] = random.randint(0, 5)
        f["posts_created"] = random.randint(3, 10)
        f["comments_written"] = random.randint(4, 15)
        f["upvotes_received"] = random.randint(5, 15)
        f["quizzes_completed"] = random.randint(2, 3)
        f["buddies_messaged"] = random.randint(4, 10)
        f["spam"] = False

    elif box == "comeback":
        f["login_streak"] = 1
        f["posts_created"] = random.randint(2, 10)
        f["karma_earned"] = random.randint(10, MAX_KARMA)
        f["karma_spent"] = random.randint(0, f["karma_earned"] - MIN_KARMA)
        # Other features low
        f["comments_written"] = random.randint(1, 5)
        f["upvotes_received"] = random.randint(0, 5)
        f["quizzes_completed"] = random.randint(1, 2)
        f["buddies_messaged"] = random.randint(0, 5)
        f["spam"] = False

    elif box == "mystery":
        f["karma_earned"] = random.randint(10, MAX_KARMA)
        f["posts_created"] = random.randint(1, 10)
        f["karma_spent"] = random.randint(0, f["karma_earned"] - MIN_KARMA)
        # Other features random moderate
        f["login_streak"] = random.randint(0, 5)
        f["comments_written"] = random.randint(0, 10)
        f["upvotes_received"] = random.randint(0, 10)
        f["quizzes_completed"] = random.randint(0, 3)
        f["buddies_messaged"] = random.randint(0, 7)
        f["spam"] = False

    elif (force_spam==True):
        # No box unlocked: set low or random features that fail all rules
        f["login_streak"] = random.randint(0, 10)
        f["posts_created"] = random.randint(0, 15)
        f["comments_written"] = random.randint(0, 20)
        f["upvotes_received"] = random.randint(0, 25)
        f["quizzes_completed"] = random.randint(0, 8)
        f["buddies_messaged"] = random.randint(0, 15)
        f["karma_earned"] = random.randint(0, 30)
        f["karma_spent"] = random.randint(0,f["karma_earned"])
        f["spam"] = True
    
    else:
        # No box unlocked: set low or random features that fail all rules
        f["login_streak"] = random.randint(0, 2)
        f["posts_created"] = random.randint(0, 2)
        f["comments_written"] = random.randint(0, 4)
        f["upvotes_received"] = random.randint(0, 5)
        f["quizzes_completed"] = random.randint(0, 0)
        f["buddies_messaged"] = random.randint(0, 2)
        f["spam"] = False

    # If forcing low karma difference (karma_earned - karma_spent < 5)
    if force_low_karma:
        f["karma_earned"] = random.randint(5, 10)
        f["karma_spent"] = f["karma_earned"] - random.randint(0, 4)
        if f["karma_spent"] < 0:
            f["karma_spent"] = 0

    # Sanity check: karma_spent should not exceed karma_earned
    if f["karma_spent"] > f["karma_earned"]:
        f["karma_spent"] = f["karma_earned"]


    return f

def generate_data(n=1000):
    data_rows = []

    n_spam = int(n * SPAM_RATIO)
    n_low_karma_diff = int(n * LOW_KARMA_DIFF_RATIO)
    n_reward_true = n - n_spam - n_low_karma_diff

    # Generate spam rows
    for _ in range(n_spam):
        features = generate_features_for_box(force_spam=True)
        row = features.copy()
        row["surprise_unlocked"] = False
        row["reward_karma"]=0
        row["reason"] = "spam"
        row["rarity"] = ""
        row["box_type"] = ""
        row["status"] = "missed"
        data_rows.append(row)

    # Generate low karma difference rows (false unlock)
    for _ in range(n_low_karma_diff):
        # Try to generate features that would unlock a box but with low karma difference
        # So surprise_box_unlocked = False, but box type guessed
        box = ""
        features = generate_features_for_box(box=box, force_low_karma=True)
        row = features.copy()
        karma = features["karma_earned"] - features["karma_spent"]
        # Check if rules pass or fail due to low karma difference - for dataset label it's false unlock
        surprise_unlocked = False
        row["surprise_unlocked"] = surprise_unlocked
        row["reward_karma"]=karma
        row["reason"] = "low_karma_diff"
        row["rarity"] = ""  # No rarity if not unlocked
        row["box_type"] =""
        row["status"] = "missed"
        data_rows.append(row)

    # Generate true reward unlock rows
    for _ in range(n_reward_true):
        box = random.choice(list(reward_rules.keys()))
        features = generate_features_for_box(box=box, force_low_karma=False)
        karma = features["karma_earned"] - features["karma_spent"]
        # If karma difference too low, increase karma_earned
        if karma < MIN_KARMA:
            features["karma_earned"] = features["karma_spent"] + MIN_KARMA + random.randint(0, 10)
        # Decide box unlock true
        surprise_unlocked = True
        rarity = determine_rarity(features["karma_earned"] - features["karma_spent"], box, features)
        reason = generate_reason(features, box)
        status = generate_status(surprise_unlocked)

        row = features.copy()
        row["surprise_unlocked"] = surprise_unlocked
        row["reward_karma"]=karma
        row["reason"] = reason
        row["rarity"] = rarity
        row["box_type"] = box
        row["status"] = status

        data_rows.append(row)

    # Shuffle rows for randomness
    random.shuffle(data_rows)
    return pd.DataFrame(data_rows)

# Example usage
if __name__ == "__main__":
    df = generate_data(n=8000)
    print(df.head())
    df.to_csv("karma_surprise_box_dataset.csv", index=False)
