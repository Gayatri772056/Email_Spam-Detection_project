
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

# Email dataset link
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep='\t', names=['label','message'])
df['label'] = df['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = round(accuracy_score(y_test, y_pred)*100,2)
precision = round(precision_score(y_test, y_pred)*100,2)
recall = round(recall_score(y_test, y_pred)*100,2)
f1 = round(f1_score(y_test, y_pred)*100,2)

cm = confusion_matrix(y_test, y_pred).tolist()

@app.route("/")
def home():
    return render_template("index.html",
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1,
                           cm=cm)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]

    confidence = round(max(prob)*100,2)

    result = "Spam 🚨" if prediction==1 else "Not Spam ✅"

    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
