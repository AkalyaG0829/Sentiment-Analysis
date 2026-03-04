import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df['label'] = df['label'].map({"positive": 1, "negative": 0})
    df = df.dropna(subset=['text', 'label'])
    return df

def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df['text'], df['label'], test_size=test_size, random_state=random_state)

def build_vectorizer():
    return TfidfVectorizer(max_features=5000, stop_words='english')

def fit_transform(vectorizer, X_train, X_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec
if __name__ == "__main__":
    df = load_data("../data/raw/imdb.csv")
    print("Dataset loaded. Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(3)) 
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nTrain size:", len(X_train), "Test size:", len(X_test))
    vectorizer = build_vectorizer()
    X_train_vec, X_test_vec = fit_transform(vectorizer, X_train, X_test)
    print("\nVectorized shapes:")
    print("X_train_vec:", X_train_vec.shape)
    print("X_test_vec:", X_test_vec.shape)