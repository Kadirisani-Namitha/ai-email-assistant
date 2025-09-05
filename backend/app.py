# app.py (CSV-only demo with Top-6 AI drafts, Intents+Requirements, RAG)

import re
import os
from datetime import datetime, timedelta
from dateutil import parser as dateparser

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

# ---------- LOAD ENV ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please add it to .env")
    st.stop()

# ---------- CONFIG ----------
CSV_PATH = "data/urgent_emails.csv"

SUPPORT_KEYWORDS = ["support", "help", "query", "request", "issue", "problem", "question", "ticket"]
URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "critical", "downtime", "blocked", "cannot access",
    "can't access", "failed", "failure", "error", "not working", "unable to log in",
    "payment failed", "escalate", "priority", "outage", "crash", "deadline"
]

INTENT_RULES = {
    "Login/Access Issue": ["login", "log in", "password", "reset", "access", "blocked", "2fa", "otp"],
    "Billing/Pricing": ["billing", "charge", "invoice", "payment", "refund", "pricing"],
    "Integration/API": ["api", "webhook", "integration", "sdk", "token", "key"],
    "Performance/Outage": ["downtime", "slow", "latency", "outage", "crash"],
    "Account/Verification": ["verify", "verification", "account", "profile", "kyc"],
    "General Query": ["query", "question", "info", "information", "details"]
}

KB = [
    ("Password reset", "To reset your password, use the 'Forgot Password' link and check your email inbox and spam."),
    ("2FA issues", "If 2FA fails, sync your device time and try backup codes from your security settings."),
    ("Pricing & Billing", "Our Pro plan is billed monthly. Invoices are available in the Billing > Invoices page."),
    ("API keys", "Generate API keys in Developer Settings. Keep them secret and rotate periodically."),
    ("Downtime/outage", "Check status page for incidents. If affected, our team will post updates within 15 minutes.")
]

# ---------- HELPERS ----------
def to_text(x): return x if isinstance(x, str) else ""

def contains_keywords(text, keywords):
    text = text.lower()
    return any(k in text for k in keywords)

def detect_intents_and_requirements(text):
    text_low = text.lower()
    intents, requirements = [], []

    for intent, words in INTENT_RULES.items():
        if any(w in text_low for w in words):
            intents.append(intent)
    if not intents:
        intents = ["General Query"]

    # Requirements (rule-based)
    if "immediate" in text_low or "urgent" in text_low or "asap" in text_low:
        requirements.append("Immediate attention required")
    if "refund" in text_low or "charged twice" in text_low:
        requirements.append("Billing correction/refund")
    if "integration" in text_low or "api" in text_low:
        requirements.append("Integration request")
    if "cannot login" in text_low or "access blocked" in text_low:
        requirements.append("Restore access")
    if not requirements:
        requirements.append("General assistance")

    return intents, requirements

def extract_emails(text): return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", to_text(text))
def extract_phone(text):
    phones = re.findall(r"(?:\+?\d[\d\-\s()]{6,}\d)", to_text(text))
    return [re.sub(r"\s+", " ", p).strip() for p in phones]

def normalize_datetime(dt):
    if isinstance(dt, str):
        try: return dateparser.parse(dt)
        except Exception: return None
    return dt

def compute_sentiment(df, text_col="body"):
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
    cats = scores.apply(lambda c: "Positive" if c >= 0.3 else ("Negative" if c <= -0.3 else "Neutral"))
    return scores, cats

def compute_priority(row):
    subj, body = to_text(row.get("subject")), to_text(row.get("body"))
    text = (subj + " " + body).lower()
    priority = 0
    if contains_keywords(text, URGENCY_KEYWORDS): priority += 2
    if row.get("sentiment") == "Negative": priority += 1
    dt = normalize_datetime(row.get("sent_date"))
    if dt and datetime.now() - dt <= timedelta(hours=24): priority += 1
    return priority

# ---------- KB + RAG ----------
def build_kb_index():
    docs = [item[1] for item in KB]
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(docs)
    return tfidf, X
TFIDF, KBX = build_kb_index()

def retrieve_kb_snippets(query, top_k=3, max_len=300):
    docs = [item[1] for item in KB]
    qv = TFIDF.transform([query])
    sims = cosine_similarity(qv, KBX).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    snippets = [docs[i] for i in idxs]
    return [s if len(s) <= max_len else s[:max_len] + "..." for s in snippets]

# ---------- OPENAI REPLY ----------
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_reply(row, intents, requirements, kb_snippets):
    sender, subject, body = row.get("sender", ""), row.get("subject", ""), to_text(row.get("body"))
    sentiment, priority = row.get("sentiment", "Neutral"), row.get("priority", 0)

    prompt = f"""
You are a helpful support assistant. Reply to the following email professionally and empathetically.

From: {sender}
Subject: {subject}
Body: {body}

Detected Sentiment: {sentiment}
Priority Score: {priority}
Intents: {', '.join(intents)}
Requirements: {', '.join(requirements)}

Knowledge Base Context:
{chr(10).join(kb_snippets)}

Draft a short, polite, professional response that addresses the issue directly.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a polite and professional support assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return f"(Fallback) Hi {sender.split('@')[0].title()}, we’ve logged your issue. Our team will get back soon."

# ---------- APP ----------
st.set_page_config(page_title="AI-Powered Communication Assistant", layout="wide")
st.title("📬 AI-Powered Communication Assistant (CSV Demo)")

df = pd.read_csv(CSV_PATH).dropna(subset=["sender", "subject", "body", "sent_date"])

# Filter support emails
mask_support = (
    df["subject"].fillna("").str.contains("|".join(SUPPORT_KEYWORDS), case=False, na=False) |
    df["body"].fillna("").str.contains("|".join(SUPPORT_KEYWORDS), case=False, na=False)
)
support_df = df[mask_support].copy()

# Enrich dataset
scores, cats = compute_sentiment(support_df, text_col="body")
support_df["sentiment_score"], support_df["sentiment"] = scores, cats
support_df[["intents", "requirements"]] = support_df["body"].apply(detect_intents_and_requirements).apply(pd.Series)
support_df["found_emails"] = support_df["body"].apply(extract_emails)
support_df["found_phones"] = support_df["body"].apply(extract_phone)
support_df["is_urgent"] = support_df.apply(
    lambda r: contains_keywords(to_text(r.subject) + " " + to_text(r.body), URGENCY_KEYWORDS), axis=1
)
support_df["priority"] = support_df.apply(compute_priority, axis=1)
support_df["queue_rank"] = support_df["priority"].rank(method="first", ascending=False).astype(int)
support_df["dt"] = support_df["sent_date"].apply(normalize_datetime)

# ---------- AI Drafts: Only Top 6 ----------
support_df = support_df.sort_values("priority", ascending=False)
support_df["ai_draft_reply"] = "[Draft reply not generated]"

for idx, row in support_df.head(6).iterrows():
    kb_snips = retrieve_kb_snippets(row["body"], top_k=3)
    reply = generate_reply(row, row["intents"], row["requirements"], kb_snips)
    support_df.at[idx, "ai_draft_reply"] = reply

# ---------- UI Filters ----------
st.sidebar.header("Filters")
sentiment_pick = st.sidebar.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive","Neutral","Negative"])
urgent_toggle = st.sidebar.selectbox("Urgency", ["All", "Urgent only", "Not urgent"])
search_text = st.sidebar.text_input("Search (subject/body/sender)")
last_hours = st.sidebar.slider("Received within last N hours", 0, 168, 0)

view_df = support_df.copy()
if sentiment_pick: view_df = view_df[view_df["sentiment"].isin(sentiment_pick)]
if urgent_toggle == "Urgent only": view_df = view_df[view_df["is_urgent"] == True]
elif urgent_toggle == "Not urgent": view_df = view_df[view_df["is_urgent"] == False]
if search_text:
    s = search_text.lower()
    view_df = view_df[
        view_df["subject"].str.lower().str.contains(s, na=False) |
        view_df["body"].str.lower().str.contains(s, na=False) |
        view_df["sender"].str.lower().str.contains(s, na=False)
    ]
if last_hours and last_hours > 0:
    cutoff = datetime.now() - timedelta(hours=last_hours)
    view_df = view_df[view_df["dt"].notna() & (view_df["dt"] >= cutoff)]

# ---------- METRICS ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Support Emails", len(support_df))
col2.metric("Urgent", int(support_df["is_urgent"].sum()))
col3.metric("Pending (unreplied)", len(support_df))
last24 = support_df[support_df["dt"].notna() & (support_df["dt"] >= datetime.now() - timedelta(hours=24))]
col4.metric("Last 24 hours", len(last24))

# ---------- VISUALS ----------
left, right = st.columns(2)
with left:
    st.subheader("Sentiment Distribution")
    st.bar_chart(support_df["sentiment"].value_counts())
with right:
    st.subheader("Urgency Breakdown")
    urgent_df = pd.DataFrame({
        "label": ["Not urgent", "Urgent"],
        "count": [int((~support_df["is_urgent"]).sum()), int(support_df["is_urgent"].sum())]
    }).set_index("label")
    st.bar_chart(urgent_df)

# ---------- PRIORITY QUEUE ----------
st.subheader("📥 Priority Queue (highest first)")

# Add urgency icons
view_df["urgency_label"] = view_df["is_urgent"].apply(lambda x: "🔴 Urgent" if x else "🟢 Not urgent")

queue_view = view_df.sort_values(["priority", "dt"], ascending=[False, False])[[
    "queue_rank","priority","urgency_label","sentiment","sender","subject","sent_date","intents","found_emails","found_phones"
]]

st.dataframe(queue_view, use_container_width=True, height=300)

# ---------- DETAIL PANEL ----------
st.subheader("🔎 Email Detail & Draft Reply")
if not queue_view.empty:
    options = {i: f"{support_df.loc[i, 'subject']} ({support_df.loc[i, 'sender']})" for i in queue_view.index}
    idx = st.selectbox("Pick an email", options.keys(), format_func=lambda i: options[i])
    row = support_df.loc[idx]

    with st.expander("Raw Email"):
        st.write("**Sender:**", row["sender"])
        st.write("**Subject:**", row["subject"])
        st.write("**Received:**", row["sent_date"])
        st.write("**Body:**"); st.write(row["body"])

    with st.expander("Extracted Info"):
        st.write("**Sentiment:**", row["sentiment"], f"({row['sentiment_score']:.2f})")
        st.write("**Urgent:**", bool(row["is_urgent"]))
        st.write("**Priority Score:**", int(row["priority"]))
        st.write("**Intents:**", ", ".join(row["intents"]))
        st.write("**Requirements:**", ", ".join(row["requirements"]))
        st.write("**Emails Found:**", ", ".join(row["found_emails"]) if row["found_emails"] else "—")
        st.write("**Phones Found:**", ", ".join(row["found_phones"]) if row["found_phones"] else "—")

    st.markdown("### ✍️ AI-Generated Draft (editable)")
    draft = st.text_area("Draft Reply", value=row["ai_draft_reply"], height=300)

    colA, colB = st.columns(2)
    with colA:
        if st.button("✅ Mark as Resolved (Demo)"):
            st.success("Marked as resolved (demo only)")
    with colB:
        if st.button("📤 Send Email (Demo)"):
            st.info("Demo only — would send via SMTP/Gmail API")
else:
    st.warning("⚠️ No emails match the current filters.")

# ---------- EXPORT ----------
st.download_button(
    label="⬇️ Download filtered results (CSV)",
    data=view_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_support_emails.csv",
    mime="text/csv",
)
