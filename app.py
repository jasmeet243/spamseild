"""SpamShield - SMS/Email Spam & Scam Detector.   Run:  streamlit run app.py"""
import html, os, time
import pandas as pd
import streamlit as st
from predict import analyse
import inbox

st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="wide")
st.title("🛡️ SpamShield")
st.caption("SMS / Email Spam & Scam Detector  |  TF-IDF + Logistic Regression + scam-tactic & URL analysis")

SAMPLES = {
    "Bank KYC scam": "Dear SBI customer your KYC is expired, account will be blocked within 24 hours. Click http://sbi-secure-login.xyz/verify to update",
    "Lottery spam": "WINNER!! You have won a £1000 cash prize. Call 09061701461 now to claim. Valid 12 hours only",
    "Job scam": "Earn Rs 5000 daily working from home. No experience needed. WhatsApp 9876543210 to join now",
    "Delivery scam": "Your parcel is held at customs. Pay Rs 49 fee at bit.ly/xk29 within 24 hours or it will be returned",
    "Obfuscated spam": "Cl@im your fr33 pr!ze n0w, c@ll 09061701461 to w1n cash",
    "Normal message": "Hey, are we meeting at 5pm for the project discussion?",
}

def render_analysis(r):
    color = {"HIGH": "#d62728", "MEDIUM": "#ff9f1c", "LOW": "#2ca02c"}[r["level"]]
    verdict = {"HIGH": "🚨 Likely SPAM / SCAM", "MEDIUM": "⚠️ Suspicious - be careful", "LOW": "✅ Looks safe"}[r["level"]]

    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"<div style='border:3px solid {color};border-radius:12px;padding:16px;text-align:center'>"
                    f"<div style='font-size:14px;color:gray'>RISK SCORE</div>"
                    f"<div style='font-size:56px;font-weight:700;color:{color}'>{r['risk']:.0f}</div>"
                    f"<div style='font-size:18px;color:{color}'>{r['level']}</div></div>", unsafe_allow_html=True)
        st.progress(min(int(r["risk"]), 100))
        st.write(f"**{verdict}**")
        st.caption(f"ML spam probability: {r['ml_prob']:.0%}  |  Rule score: {r['rule_score']}/100  \n"
                   f"Risk = 0.7 × ML probability + 0.3 × rule score" + (" (ML weight reduced: looks like a genuine OTP warning)" if r["legit_warning"] else ""))
    with right:
        for f in r["red_flags"]:
            st.error("🚩 " + f)
        st.subheader("Why was it flagged?")
        spans = []
        for w, c in r["contrib"]:
            w_html = html.escape(w)
            if c > 0.8:
                spans.append(f"<span style='background:#ffb3b3;padding:1px 4px;border-radius:4px'>{w_html}</span>")
            elif c < -0.8:
                spans.append(f"<span style='background:#b9f0c2;padding:1px 4px;border-radius:4px'>{w_html}</span>")
            else:
                spans.append(w_html)
        st.markdown(" ".join(spans), unsafe_allow_html=True)
        st.caption("🟥 pushes towards spam   🟩 pushes towards genuine  (word weights = Logistic Regression coefficients)")
        if r["top_words"]:
            st.write("**Top spam-triggering words:** " + ", ".join(f"`{w}` ({c:+.1f})" for w, c in r["top_words"]))

    st.divider()
    a, b = st.columns(2)
    with a:
        st.subheader("🧠 Scam tactics detected")
        any_t = False
        for name, hits in r["tactics"].items():
            level = "High" if len(hits) >= 3 else "Medium" if len(hits) == 2 else "Low" if hits else "None"
            any_t |= bool(hits)
            st.write(f"**{name}:** {level}" + (f"  — {', '.join(hits)}" if hits else ""))
            st.progress(min(len(hits), 3) / 3)
        if not any_t:
            st.info("No manipulation tactics found.")
    with b:
        st.subheader("🔗 Links & phone numbers")
        if r["urls"]:
            for u in r["urls"]:
                st.write(f"**{u['url']}**")
                if u["flags"]:
                    for fl in u["flags"]:
                        st.warning(fl)
                else:
                    st.success("No suspicious signals")
        else:
            st.info("No links found.")
        if r["phones"]:
            st.write("**Phone numbers found:** " + ", ".join(r["phones"]))
            st.caption("Never call numbers from unsolicited messages; verify on the company's official site.")
    with st.expander("Preprocessed text (what the model actually sees)"):
        st.code(r["cleaned"])


tab1, tab3, tab2 = st.tabs(["🔍 Check a message", "📥 Inbox scanner", "📊 Model results"])

with tab1:
    st.write("**Try a sample:**")
    cols = st.columns(len(SAMPLES))
    for c, (name, txt) in zip(cols, SAMPLES.items()):
        if c.button(name):
            st.session_state["msg"] = txt
    msg = st.text_area("Paste an SMS or email here", key="msg", height=130)

    if st.button("Analyse", type="primary") and msg.strip():
        render_analysis(analyse(msg))

with tab2:
    st.subheader("Model comparison (20% held-out test set)")
    if os.path.exists("results/metrics.csv"):
        st.dataframe(pd.read_csv("results/metrics.csv"), hide_index=True)
        c1, c2 = st.columns(2)
        c1.image("results/confusion_matrix.png", caption="Confusion matrix - Logistic Regression")
        c2.image("results/top_words.png", caption="Most influential words")
        st.subheader("Robustness against obfuscated spam (fr33, w1n, pr!ze)")
        st.dataframe(pd.read_csv("results/robustness.csv"), hide_index=True)
    else:
        st.warning("Run `python train.py` first.")

with tab3:
    st.subheader("Scan your emails and SMS automatically")
    st.caption("Read-only: SpamShield never marks, moves or deletes anything, and nothing is stored. "
               "Note: the model was trained on SMS, so for emails it reads the wording while separate rules judge the links.")
    mode = st.radio("Source", ["📧 Connect my mailbox (IMAP)", "📁 Upload files (emails / SMS export)"], horizontal=True)

    def show_results(rows):
        if not rows:
            st.info("No messages found."); return
        hi = sum(r["result"]["level"] == "HIGH" for r in rows); me = sum(r["result"]["level"] == "MEDIUM" for r in rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scanned", len(rows)); c2.metric("🚨 High risk", hi); c3.metric("⚠️ Medium", me); c4.metric("✅ Low", len(rows) - hi - me)
        icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
        df = pd.DataFrame([{"Level": icon[r["result"]["level"]] + " " + r["result"]["level"], "Risk": round(r["result"]["risk"]),
                            "Type": r["msg"]["source"], "From": r["msg"]["sender"][:40], "Message": r["preview"],
                            "Why": r["reason"][:90], "Date": r["msg"]["date"]} for r in rows])
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode("utf-8"), "spamshield_scan.csv", "text/csv")
        pick = st.selectbox("Inspect a message in detail", range(len(rows)), format_func=lambda i: f"{df.iloc[i]['Level']}  {df.iloc[i]['Risk']:>3}  |  {df.iloc[i]['Message']}")
        m = rows[pick]["msg"]
        with st.expander("Full message text", expanded=False):
            st.write(f"**From:** {m['sender']}  \n**Subject:** {m['subject']}")
            st.text(m["body"][:1500])
        render_analysis(rows[pick]["result"])

    if mode.startswith("📧"):
        with st.expander("How to connect Gmail (use an App Password, never your real password)"):
            st.markdown("1. Turn on **2-Step Verification** in your Google account.\n"
                        "2. Search **App passwords** in your Google account, create one named *SpamShield*, copy the 16-letter code.\n"
                        "3. Use `imap.gmail.com`, your Gmail address and that code below. Outlook: `outlook.office365.com`.\n\n"
                        "Your details stay only in this browser session and are never saved.")
        c1, c2 = st.columns(2)
        host = c1.text_input("IMAP server", "imap.gmail.com"); user = c2.text_input("Email address")
        pw = c1.text_input("App password", type="password"); folder = c2.text_input("Folder", "INBOX")
        n = c1.slider("How many recent emails", 5, 100, 20); unseen = c2.checkbox("Only unread emails")
        auto = st.checkbox("🔄 Auto-scan on its own"); mins = st.slider("Every how many minutes", 1, 30, 5) if auto else 5

        if st.button("Scan now", type="primary"):
            st.session_state["scan_now"] = True

        def imap_panel():
            requested = st.session_state.pop("scan_now", False)
            auto_due = auto and st.session_state.get("last_scan") and time.time() - st.session_state["last_scan"] >= mins * 60 - 1
            if requested and not (user and pw):
                st.warning("Enter your email address and App Password first.")
            elif requested or auto_due:
                try:
                    with st.spinner("Reading your mailbox..."):
                        st.session_state["rows"] = inbox.scan(inbox.fetch_imap(host, user, pw, folder, n, unseen))
                    st.session_state["last_scan"] = time.time()
                except Exception as e:
                    st.error(f"Could not read the mailbox: {e}. Check the server, App Password and that IMAP is enabled.")
            if st.session_state.get("last_scan"):
                st.caption("Last scan: " + time.strftime("%H:%M:%S", time.localtime(st.session_state["last_scan"]))
                           + (f"  |  next auto-scan in about {mins} min" if auto else ""))
            if st.session_state.get("rows") is not None:
                show_results(st.session_state["rows"])

        if auto and hasattr(st, "fragment"):
            st.fragment(run_every=f"{mins * 60}s")(imap_panel)()
        else:
            imap_panel()
    else:
        st.write("Upload emails (**.eml**, **.mbox** from Google Takeout) or an SMS export "
                 "(**.xml** from the *SMS Backup & Restore* app, **.csv**, or **.txt** with one message per line).")
        ups = st.file_uploader("Drop files here", type=["eml", "mbox", "xml", "csv", "txt"], accept_multiple_files=True)
        if ups:
            try:
                show_results(inbox.scan(inbox.parse_uploads([(f.name, f.getvalue()) for f in ups])))
            except Exception as e:
                st.error(f"Could not read these files: {e}")
