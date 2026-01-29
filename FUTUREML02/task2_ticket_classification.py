import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv(
    "all_tickets_processed_improved_v3.csv",
    encoding="latin1"
)

print(data.head())

X = data["Document"]
y = data["Topic_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", LinearSVC())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nModel Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

new_ticket = [
    "VPN connection fails after password reset"
]

predicted_topic = model.predict(new_ticket)[0]

print("\nNew Ticket Prediction")
print("Ticket:", new_ticket[0])
print("Predicted Category:", predicted_topic)
