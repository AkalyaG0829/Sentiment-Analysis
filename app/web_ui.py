from flask import Flask, render_template, request
import joblib
app=Flask(__name__)
model = joblib.load("../models/sentiment_model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
@app.route("/", methods=["GET", "POST"])
def home():
    review = ""
    sentiment = ""
    if request.method=="POST":
        review=request.form["review"]
        review_vec=vectorizer.transform([review])
        prediction=model.predict(review_vec)[0]
        sentiment="Positive" if prediction == 1 else "Negative"
    return render_template("index.html", review=review, sentiment=sentiment)

if __name__=="__main__":
    app.run(debug=True)