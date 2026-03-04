from flask import Flask, request, jsonify
import joblib
model=joblib.load("../models/sentiment_model.pkl")
vectorizer=joblib.load("../models/vectorizer.pkl")
app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    data=request.get_json()
    review=data.get("review", "")
    X=vectorizer.transform([review])
    prediction=model.predict(X)[0]
    sentiment="positive" if prediction==1 else "negative"
    return jsonify({"review": review, "sentiment": sentiment})
if __name__=="__main__":
    app.run(debug=True)