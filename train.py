"""Train + compare Naive Bayes, Logistic Regression, SVM on the SMS Spam Collection.
Run:  python train.py
Outputs: models/spamshield.joblib, results/metrics.csv, results/confusion_matrix.png, results/top_words.png
"""
import os, time
import joblib, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from preprocess import clean

os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)

# 1. Load data ------------------------------------------------------------
df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "text"])
df = df.drop_duplicates().reset_index(drop=True)
df["y"] = (df["label"] == "spam").astype(int)
print(f"Messages: {len(df)} | spam: {df.y.sum()} ({df.y.mean():.1%}) | ham: {(df.y==0).sum()}")

# 2. Preprocess -----------------------------------------------------------
df["clean"] = df["text"].apply(clean)

# 3. Split (stratified keeps the spam ratio equal in train and test) ------
X_tr, X_te, y_tr, y_te, raw_tr, raw_te = train_test_split(
    df["clean"], df["y"], df["text"], test_size=0.2, stratify=df["y"], random_state=42)

# 4. TF-IDF with unigrams + bigrams ---------------------------------------
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
print("TF-IDF features:", Xtr.shape[1])

# 5. Compare models -------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=10, class_weight="balanced", max_iter=1000),
    "Linear SVM": LinearSVC(C=1, class_weight="balanced"),
}
rows, fitted = [], {}
for name, m in models.items():
    t0 = time.time(); m.fit(Xtr, y_tr); tt = time.time() - t0
    p = m.predict(Xte)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, p), "Precision (spam)": precision_score(y_te, p),
                 "Recall (spam)": recall_score(y_te, p), "F1 (spam)": f1_score(y_te, p), "Train time (s)": round(tt, 3)})
    fitted[name] = m
res = pd.DataFrame(rows).round(4)
print("\n", res.to_string(index=False))
res.to_csv("results/metrics.csv", index=False)

# 6. Main model = Logistic Regression (gives probabilities + coefficients) -
lr = fitted["Logistic Regression"]
pred = lr.predict(Xte)
print("\nClassification report (Logistic Regression):\n", classification_report(y_te, pred, target_names=["ham", "spam"]))

cm = confusion_matrix(y_te, pred)
plt.figure(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix - Logistic Regression"); plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# top spam / ham words from coefficients
names = np.array(vec.get_feature_names_out()); coef = lr.coef_[0]
top_spam = pd.Series(coef, index=names).nlargest(15); top_ham = pd.Series(coef, index=names).nsmallest(15)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
top_spam[::-1].plot.barh(ax=ax[0], color="crimson"); ax[0].set_title("Top SPAM words (positive coef.)")
top_ham[::-1].plot.barh(ax=ax[1], color="seagreen"); ax[1].set_title("Top HAM words (negative coef.)")
plt.tight_layout(); plt.savefig("results/top_words.png", dpi=150); plt.close()

# 7. Robustness test: obfuscated spam ------------------------------------
# A plain "typical" pipeline (lowercase + strip symbols) is compared with ours,
# on spam that has been disguised like  fr33 / w1n / pr!ze / 73xt.
import re
from robustness import obfuscate
from preprocess import normalise
basic = lambda t: re.sub(r"[^a-z\s]", " ", t.lower())
tr_raw = df.loc[X_tr.index, "text"]
bvec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
blr = LogisticRegression(C=10, class_weight="balanced", max_iter=1000).fit(bvec.fit_transform(tr_raw.map(basic)), y_tr)

spam_raw = raw_te[y_te == 1].tolist()
obf = [obfuscate(t, seed=i) for i, t in enumerate(spam_raw)]
def rec(model, vectorizer, texts, prep):
    return recall_score([1] * len(texts), model.predict(vectorizer.transform([prep(t) for t in texts])))
rob = pd.DataFrame([
    {"Pipeline": "Baseline (no normaliser)",  "Test set": "Original spam",   "Spam caught": rec(blr, bvec, spam_raw, basic)},
    {"Pipeline": "Baseline (no normaliser)",  "Test set": "Obfuscated spam", "Spam caught": rec(blr, bvec, obf, basic)},
    {"Pipeline": "Baseline + normaliser",     "Test set": "Obfuscated spam", "Spam caught": rec(blr, bvec, obf, lambda t: basic(normalise(t)))},
    {"Pipeline": "SpamShield (full pipeline)", "Test set": "Original spam",   "Spam caught": rec(lr, vec, spam_raw, clean)},
    {"Pipeline": "SpamShield (full pipeline)", "Test set": "Obfuscated spam", "Spam caught": rec(lr, vec, obf, clean)},
]).round(4)
print("\nRobustness test:\n", rob.to_string(index=False)); rob.to_csv("results/robustness.csv", index=False)
for i in range(2): print("\n  original  :", spam_raw[i][:100], "\n  obfuscated:", obf[i][:100])

# false-alarm check: normaliser must not damage normal (ham) messages
ham_raw = raw_te[y_te == 0].tolist()
fp_with = (lr.predict(vec.transform([clean(t) for t in ham_raw])) == 1).mean()
fp_without = (lr.predict(vec.transform([clean(t, use_normalise=False) for t in ham_raw])) == 1).mean()
print(f"\nFalse-positive rate on ham  with normaliser: {fp_with:.4f} | without: {fp_without:.4f}")

# 8. Save -----------------------------------------------------------------
joblib.dump({"vectorizer": vec, "model": lr}, "models/spamshield.joblib")
print("\nSaved models/spamshield.joblib")

# false positives / negatives for error analysis
err = pd.DataFrame({"text": raw_te.values, "actual": y_te.values, "pred": pred})
err[err.actual != err.pred].to_csv("results/errors.csv", index=False)
