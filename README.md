# SpamShield – SMS/Email Spam & Scam Detector

**Course / Subject:** Natural Language Processing (CA Project)
**Student Name:** _______________  **Roll No.:** _______________
**Institute:** _______________  **Submitted to:** _______________

---

## 1. Introduction

Spam and scam messages (fake lotteries, "KYC expired" bank messages, OTP fraud, fake delivery fees, job scams) are a daily problem, especially on mobile. Most spam filters only output a label such as "spam" or "not spam", which does not help a user understand *why* a message is dangerous.

**SpamShield** is an NLP-based detector that classifies an SMS/email as spam or genuine and also **explains its decision**. It combines a machine-learning text classifier with rule-based scam analysis (psychological tactics and suspicious links) and presents the result as a 0–100 risk score in a web app.

## 2. Objectives

1. Build an accurate spam classifier using classical NLP techniques (TF-IDF + Logistic Regression).
2. Compare it against other models (Naive Bayes, Linear SVM) using proper metrics for imbalanced data.
3. Make the system explainable: show which words and which signals caused a message to be flagged.
4. Detect common scam manipulation tactics and suspicious URLs/phone numbers.
5. Test and improve robustness against deliberately disguised text (e.g. `fr33 pr!ze`).
6. Deliver everything in a simple, usable web interface.

## 3. What makes this project different

| Feature | Description | File |
|---|---|---|
| **Risk score (0–100)** | `Risk = 0.7 × ML probability + 0.3 × rule score`, shown as Low / Medium / High | `predict.py` |
| **Word-level explanation** | Each word is coloured by its Logistic Regression coefficient (red = pushes towards spam, green = towards genuine) | `predict.py`, `app.py` |
| **Scam-tactic detection** | Detects Urgency, Fear/Threat, Greed/Reward, Authority impersonation and Action-bait using keyword dictionaries | `features.py` |
| **URL & phone analysis** | Flags URL shorteners, suspicious TLDs (.xyz, .top…), IP-based links, many hyphens, "@" tricks, non-HTTPS, lookalike bank/brand domains | `features.py` |
| **Red-flag overrides** | Requests to share OTP/PIN/CVV/password, or links with 3+ suspicious signals, force a High score | `features.py` |
| **Anti-obfuscation normaliser** | Converts `fr33 pr!ze 73xt w1n` → `free prize text win` before classification | `preprocess.py` |
| **Inbox scanner** | Connects to a mailbox (IMAP, read-only) or reads uploaded emails/SMS exports and rates every message automatically, sorted by risk | `inbox.py`, `app.py` |
| **Robustness experiment** | Simulates spammer evasion and measures the improvement | `robustness.py`, `train.py` |

## 4. Dataset

- **Name:** SMS Spam Collection (UCI / Kaggle)
- **Size:** 5,574 messages (5,169 after removing duplicates)
- **Classes:** 653 spam (12.6%) and 4,516 ham/genuine (87.4%) — an **imbalanced** dataset, so accuracy alone is misleading
- **Split:** 80% training (4,135) / 20% testing (1,034), stratified so both sets keep the same spam ratio
- **File:** `data/sms.tsv` (tab-separated: label, message)

## 5. Methodology

```
Raw message
   │
   ├─► URLs → "urltoken"
   ├─► Normaliser (fr33 → free, pr!ze → prize)
   ├─► Numbers → "num" / "longnum", currency symbols → "moneysym"
   ├─► Lowercase, remove punctuation, remove stopwords*, Porter stemming
   │
   ▼
TF-IDF (unigrams + bigrams, 5,915 features)
   │
   ▼
Logistic Regression ──► spam probability ─┐
                                          ├─► Risk score (0–100) ─► Low / Medium / High
Rules: tactics + URL flags + red flags ───┘
```
\*Words that carry spam signal (`free`, `now`, `call`, `you`, `your`, `urgent`, `only`) are deliberately kept.

**Key design choices**
- **TF-IDF** weights a word by how important it is to a message relative to all messages, which is better than raw word counts.
- **Bigrams** capture phrases such as "call now" or "claim prize".
- **Replacing URLs and numbers with tokens** lets the model learn "contains a link / long phone number" instead of memorising specific values.
- **`class_weight="balanced"`** penalises mistakes on the rare spam class more heavily.
- **Logistic Regression** was chosen as the main model because it provides probabilities (for the risk score) and coefficients (for explanations).

## 6. Results

### 6.1 Model comparison (20% held-out test set)

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---|---|---|---|
| Naive Bayes | 98.45% | 0.960 | 0.916 | 0.938 |
| **Logistic Regression (selected)** | **98.74%** | **0.954** | **0.947** | **0.950** |
| Linear SVM | 98.65% | 0.954 | 0.939 | 0.946 |

All three models perform similarly; Logistic Regression is slightly best on accuracy and F1. Because only ~12.6% of messages are spam, **precision, recall and F1 on the spam class** are the meaningful metrics. Plots are in `results/` (`confusion_matrix.png`, `top_words.png`).

### 6.2 Robustness against disguised spam

Spam in the test set was disguised the way real spammers do it (e.g. `fr33`, `w1n`, `pr!ze`, `73xt`).

| Pipeline | Test set | Spam caught (recall) |
|---|---|---|
| Baseline (no normaliser) | Original spam | 92.4% |
| Baseline (no normaliser) | Obfuscated spam | **69.5%** |
| Baseline + normaliser | Obfuscated spam | **87.0%** |

Disguising words reduced a plain pipeline's recall by about 23 points; the normaliser recovered most of it. It does not increase false alarms on genuine messages (false-positive rate 0.66% with the normaliser vs 0.78% without).

## 7. Project structure

```
spamshield/
├── app.py              # Streamlit web interface
├── predict.py          # Inference: risk score, explanation, combines all components
├── preprocess.py       # Text cleaning + anti-obfuscation normaliser
├── features.py         # Scam-tactic detection, URL/phone analysis, red-flag rules
├── inbox.py            # Inbox scanner: IMAP, .eml/.mbox, SMS .xml/.csv/.txt parsing + batch scoring
├── robustness.py       # Simulates obfuscation attacks used in testing
├── train.py            # Trains/compares models, generates plots and metrics
├── requirements.txt    # Python dependencies
├── data/sms.tsv        # SMS Spam Collection dataset
├── models/             # Saved model (spamshield.joblib)
└── results/            # metrics.csv, robustness.csv, confusion_matrix.png, top_words.png, errors.csv
```

## 8. How to run

**Requirements:** Python 3.9 or newer.

```powershell
# 1. Open a terminal inside the project folder (the one containing train.py)
python -m pip install -r requirements.txt

# 2. Train the models (about 10 seconds) – creates models/ and results/
python train.py

# 3. Launch the web app (opens at http://localhost:8501)
python -m streamlit run app.py
```

**Using the app:** click a sample message or paste your own SMS/email, then press **Analyse**. The app shows the risk score, the highlighted words that influenced the decision, the scam tactics found, the analysis of any links/phone numbers, and the preprocessed text. The **Model results** tab shows the comparison table, confusion matrix and robustness results.

### 8.1 Inbox scanner (optional feature)

The **Inbox scanner** tab scans many messages at once and lists them by risk (High / Medium / Low) with a short reason; any message can be opened for the full explanation, and results can be downloaded as CSV.

- **Email by IMAP:** enter the IMAP server (e.g. `imap.gmail.com`), your address and a Gmail **App Password** (create one under Google Account > Security > 2-Step Verification > App passwords). The mailbox is opened read-only, so nothing is marked, moved or deleted, and nothing is stored. An optional *auto-scan* re-checks every few minutes.
- **File upload:** `.eml` or `.mbox` (Google Takeout) for emails; `.xml` (Android *SMS Backup & Restore*), `.csv` or `.txt` for SMS.
- **Email-specific checks:** HTML emails are converted to text and every link is inspected. A link whose visible text shows one site but whose real address is another (e.g. text `paypal.com`, link `paypa1-secure-login.xyz`) is flagged as a red flag.
- **Why emails are handled differently:** the model is trained on SMS, where a link almost always means spam, but links are normal in emails. For emails the ML model therefore reads only the wording (links removed) and the link rules judge the links; only the first 2,000 characters are analysed.

## 9. Example outputs

| Message | Risk | Level |
|---|---|---|
| "Hey, are we meeting at 5pm for the project discussion?" | ~1 | LOW |
| "WINNER!! You have won a £1000 cash prize. Call 09061701461 now to claim." | ~77 | HIGH |
| "Dear SBI customer your KYC is expired… click http://sbi-secure-login.xyz/verify" | ~87 | HIGH (suspicious TLD, hyphens, lookalike of "sbi", non-HTTPS) |
| "Your parcel is held at customs. Pay Rs 49 fee at bit.ly/xk29 within 24 hours" | ~66 | HIGH (shortener, urgency, authority) |
| "Sir I am calling from bank, please share otp to avoid account block" | 60 | HIGH (red flag: asks for OTP) |

## 10. Limitations

- The dataset is English-only and dates from around 2012 (UK/Singapore SMS). It contains no Indian banking, UPI or OTP-style messages, which is why rule-based checks were added to cover that gap.
- Keyword dictionaries and the 0.7 / 0.3 risk weights are hand-chosen, not learned from data.
- A genuine bank OTP message can still receive a Medium score; a "do not share" heuristic only reduces the risk, it does not eliminate it.
- The normaliser cannot catch every evasion trick (e.g. Unicode look-alike characters).
- Messages in Hinglish or other languages are not supported.
- The model is trained on SMS only. Email results rely more on the rule-based checks; training on an email dataset (e.g. SpamAssassin) would improve them. SMS cannot be read directly from a phone by a web app, so an exported file is scanned instead.

## 11. Future work

- Add Hinglish / multilingual support (e.g. fine-tuning XLM-RoBERTa or MuRIL).
- Classify the *type* of scam (phishing, lottery, job, delivery, OTP fraud).
- Learn the rule weights and tactic dictionaries from labelled data.
- Add a user-feedback loop to retrain the model on reported messages.
- Compare with transformer models (DistilBERT/BERT) on accuracy vs. speed.
- Package as a Telegram bot or browser extension.

## 12. Technologies used

Python, pandas, scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, LinearSVC), NLTK (Porter stemmer), matplotlib, seaborn, joblib, Streamlit.

## 13. References

1. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* ACM Symposium on Document Engineering.
2. UCI Machine Learning Repository – SMS Spam Collection Dataset.
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
4. Streamlit documentation – https://docs.streamlit.io
