import pickle

import pandas as pd
import truststore
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


truststore.inject_into_ssl()


# Load dataset
df = pd.read_csv("data.csv")

# Basic cleaning
df = df.dropna()
df = df.drop_duplicates()

# Ensure correct format
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str).str.upper()

# Use SHORT TEXT (critical for real-world usage)
df["text"] = df["text"].str[:200]

# Convert labels to numbers
df["label_num"] = df["label"].map({"FAKE": 0, "REAL": 1})

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
print("Encoding text...")
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["label_num"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train classifier
print("Training classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Save classifier
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save embedder name
with open("embedder.txt", "w") as f:
    f.write("all-MiniLM-L6-v2")

print("training complete")