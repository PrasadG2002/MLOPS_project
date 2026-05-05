from fastapi import FastAPI
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


# Home route (fixes your 404 issue)
@app.get("/")
def home():
    return {"message": "Content Moderation API is running"}


# Prediction route
@app.post("/predict")
def predict(text: str):
    # Convert text to vector
    text_vec = vectorizer.transform([text])

    # Predict
    prediction = model.predict(text_vec)[0]

    return {
        "input": text,
        "toxic": int(prediction)
    }