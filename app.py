import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

@st.cache_resource
def load_models():
    category_model = joblib.load('models/category_model.pkl')
    priority_model = joblib.load('models/priority_model.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return category_model, priority_model, tfidf

category_model, priority_model, tfidf = load_models()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")
st.title("🎫 Support Ticket Classifier & Prioritizer")
st.write("**Future Interns ML Task 2** - Auto-categorizes tickets and sets priority.")

ticket_text = st.text_area("Paste customer ticket text:", "I was charged twice for my subscription. Please refund urgently.", height=150)

if st.button("Classify Ticket", type="primary"):
    if ticket_text:
        cleaned = clean_text(ticket_text)
        vectorized = tfidf.transform([cleaned]).toarray()
        category = category_model.predict(vectorized)[0]
        priority = priority_model.predict(vectorized)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📂 Category", category)
        with col2:
            if priority == 'High' or priority == 'Critical':
                st.error(f"🔥 Priority: {priority}")
            elif priority == 'Medium':
                st.warning(f"⚠️ Priority: {priority}")
            else:
                st.success(f"✅ Priority: {priority}")
        
        st.write(f"**Auto-routing:** Send to {category} team")
        st.write(f"**SLA:** {priority} priority")