import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('dataset/customer_support_tickets.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Text Cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print("Cleaning text...")
df['clean_description'] = df['ticket_text'].apply(clean_text)

# TF-IDF Vectorization
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['clean_description'])

# Train Category Model
print("Training category model...")
y_category = df['category']
X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
category_model = LogisticRegression(max_iter=1000)
category_model.fit(X_train, y_train)
y_pred = category_model.predict(X_test)
print("Category Accuracy:", accuracy_score(y_test, y_pred))

# Train Priority Model
print("Training priority model...")
y_priority = df['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)
priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train, y_train)
y_pred = priority_model.predict(X_test)
print("Priority Accuracy:", accuracy_score(y_test, y_pred))

# Create models folder
os.makedirs('models', exist_ok=True)

# Save models
print("Saving models...")
joblib.dump(category_model, 'models/category_model.pkl')
joblib.dump(priority_model, 'models/priority_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Priority Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('models/category_confusion_matrix.png')
print("All models saved successfully!")