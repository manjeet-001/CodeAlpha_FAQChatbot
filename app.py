mport re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQS = [
    {"question":"What are your working hours?", "answer":"Our support team is available Monday to Friday, 9:00 AM to 6:00 PM."},
    {"question":"How can I reset my password?", "answer":"Open the login page, select 'Forgot password', and follow the reset instructions."},
    {"question":"How do I create an account?", "answer":"Click 'Sign Up', enter your basic details, verify your email, and create a password."},
    {"question":"How can I contact customer support?", "answer":"You can contact support through the support email or the contact form on the website."},
    {"question":"Can I cancel my order?", "answer":"Orders can generally be cancelled before they are processed. Contact support with your order ID."},
    {"question":"How long does delivery take?", "answer":"Standard delivery usually takes 3–7 business days, depending on location."},
    {"question":"What payment methods are accepted?", "answer":"Common supported methods include cards, UPI, and online banking. Availability can vary."},
    {"question":"How do I track my order?", "answer":"Open your order history and select the order to view its tracking status."},
    {"question":"Is my account secure?", "answer":"Use a strong unique password and enable multi-factor authentication when available."},
    {"question":"How can I update my profile?", "answer":"Go to Account/Profile settings, edit your information, and save the changes."},
]

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

questions = [clean(x["question"]) for x in FAQS]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(questions)

def get_answer(user_question, threshold=0.18):
    query = vectorizer.transform([clean(user_question)])
    scores = cosine_similarity(query, matrix)[0]
    index = scores.argmax()
    score = scores[index]
    if score < threshold:
        return "Sorry, I don't have a confident answer for that question. Please contact support.", score
    return FAQS[index]["answer"], score

st.set_page_config(page_title="FAQ AI Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")
st.write("NLP-based FAQ matching using TF-IDF and cosine similarity.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user = st.chat_input("Ask a frequently asked question...")
if user:
    st.session_state.messages.append({"role":"user", "content":user})
    answer, score = get_answer(user)
    st.session_state.messages.append({"role":"assistant", "content":answer})
    with st.chat_message("user"):
        st.write(user)
    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(f"Similarity score: {score:.2f}")

with st.expander("Available FAQ topics"):
    for item in FAQS:
        st.write("•", item["question"])
