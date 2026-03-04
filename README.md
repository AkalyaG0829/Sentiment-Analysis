# Task 3:End-to-End Sentiment Analysis Project

## 📌 Overview 
It demonstrates an **end-to-end machine learning pipeline** for sentiment analysis using the **IMDb Movie Reviews dataset**.  
The solution includes **data preprocessing, model training, inference, and deployment with a Flask Web UI + API**.

---

## Tech Stack
- **Python 3.10+**
- **scikit-learn** (Logistic Regression, TF-IDF Vectorizer)
- **Flask** (Web UI + API)
- **Joblib** (Model persistence)
- **HTML/CSS** (Frontend styling)

---

## Steps Implemented

### 1️⃣ Preprocessing
- Loaded IMDb dataset (50,000 reviews).
- Split into **train (40,000)** and **test (10,000)** sets.
- Applied **TF-IDF vectorization** (max features = 5000).
- Verified vectorized shapes.

### 2️⃣ Model Training
- Trained **Logistic Regression** classifier.
- Achieved **~89% accuracy** on test set.
- Saved model + vectorizer using Joblib.

### 3️⃣ Inference
- Loaded trained model/vectorizer.
- Predicted sentiment for sample reviews: 
   Review: I really enjoyed this movie! -> Sentiment: Positive 
   Review: This was the worst film ever. -> Sentiment: Negative


### 4️⃣ Web Deployment
- Built **Flask app** with:
- **Web UI** (`index.html` + `style.css`)
- **API endpoint** for programmatic access
- User can type a review → see **Review + Predictive Statement**.

---

## 🎨 Web UI Preview
- Dark theme with glowing red accents.
- Review box styled with red glow.
- Sentiment output:
- **Positive**
- **Negative**

Example:
Review: The movie was amazing!
Predictive Statement: Positive


---

## ▶️ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run preprocessing:
   python preprocessing.py

3. Train the model:
   python train_model.py

4. Test inference:
   python inference.py

5. Launch Flask Web UI:
   python web_ui.py

6. Open browser → http://127.0.0.1:5000/



     