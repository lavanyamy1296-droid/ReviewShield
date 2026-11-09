from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing import clean_text

DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "reviews.csv"

class ReviewDetector:
    def __init__(self):
        data = pd.read_csv(DATA_PATH)
        data = data.dropna(subset=["review", "label"])
        data["review"] = data["review"].astype(str).map(clean_text)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data["review"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
        )
        self.models = {
            "Logistic Regression": Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
            ]),
            "Random Forest": Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
                ("classifier", RandomForestClassifier(n_estimators=160, random_state=42, class_weight="balanced", n_jobs=-1))
            ])
        }
        self.scores = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            pred = model.predict(self.X_test)
            self.scores[name] = {
                "Accuracy": round(accuracy_score(self.y_test, pred), 3),
                "Precision": round(precision_score(self.y_test, pred, pos_label="fake", zero_division=0), 3),
                "Recall": round(recall_score(self.y_test, pred, pos_label="fake", zero_division=0), 3),
                "F1-score": round(f1_score(self.y_test, pred, pos_label="fake", zero_division=0), 3)
            }
        self.metrics = {
            "training_rows": len(self.X_train),
            "test_rows": len(self.X_test),
            "features": len(self.models["Logistic Regression"].named_steps["tfidf"].get_feature_names_out()),
            "scores": self.scores
        }

    def predict(self, review, model_name="Logistic Regression"):
        model = self.models[model_name]
        cleaned = clean_text(review)
        probabilities = model.predict_proba([cleaned])[0]
        classes = list(model.classes_)
        fake_probability = float(probabilities[classes.index("fake")])
        label = "Potentially fake" if fake_probability >= 0.5 else "Likely genuine"
        confidence = fake_probability if label == "Potentially fake" else 1 - fake_probability
        return {"label": label, "confidence": confidence, "word_count": len(str(review).split())}

    def predict_many(self, reviews, model_name="Logistic Regression"):
        return [self.predict(review, model_name) for review in reviews]
