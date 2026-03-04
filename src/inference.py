import joblib
model=joblib.load("../models/sentiment_model.pkl")
vectorizer=joblib.load("../models/vectorizer.pkl")
def predict_sentiment(text):
    X=vectorizer.transform([text])
    prediction=model.predict(X)[0]
    return "positive" if prediction==1 else "negative"
if __name__=="__main__":
    sample_reviews=[
        "I really enjoyed this movie!",
        "This was the worst film ever."
    ]
    for review in sample_reviews:
        print(f"Review:{review} → Sentiment:{predict_sentiment(review)}")