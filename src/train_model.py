import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import load_data, split_data, build_vectorizer, fit_transform
df=load_data("../data/raw/imdb.csv")
X_train, X_test, y_train, y_test=split_data(df)
vectorizer=build_vectorizer()
X_train_vec, X_test_vec=fit_transform(vectorizer, X_train, X_test)
model=LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred=model.predict(X_test_vec)
accuracy=accuracy_score(y_test, y_pred)
print(f"✔️Model accuracy: {accuracy:.2f}")
joblib.dump(model, "../models/sentiment_model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
print("✔️Model and vectorizer saved!")