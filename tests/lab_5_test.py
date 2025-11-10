from sklearn.model_selection import train_test_split
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.tfidf_vectorizer import TfidfVectorizer
from src.models.text_classifier import TextClassifier

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
    ]

labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = RegexTokenizer()
vectorizer = TfidfVectorizer(tokenizer)

clf = TextClassifier(vectorizer)
clf.fit(train_texts, train_labels)

preds = clf.predict(test_texts)
metrics = clf.evaluate(test_labels, preds)
print("Evaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
