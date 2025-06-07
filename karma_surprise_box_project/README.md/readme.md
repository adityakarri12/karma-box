# Karma Surprise Box AI Microservice

This microservice evaluates user behavior and rewards surprise karma boxes using an AI model and reward rules.

## How to Run

```bash
pip install -r requirements.txt
python model/train_model.py
uvicorn app.main:app --reload
