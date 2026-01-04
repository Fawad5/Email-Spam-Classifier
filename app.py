
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download necessary NLTK data (if not already downloaded in the environment)
# This is crucial for the app to run independently
try:
    nltk.data.find('corpora/stopwords')
except Exception: # Changed from nltk.downloader.DownloadError
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except Exception: # Changed from nltk.downloader.DownloadError
    nltk.download('punkt')

# Load the trained model and TfidfVectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocessing function (must be the same as used during training)
def preprocess_text(text):
    # 1. Lower case
    text = text.lower()

    # 2. Tokenization
    tokens = nltk.word_tokenize(text)

    # 3. Remove special characters and apply stemming
    ps = PorterStemmer()
    processed_tokens = []
    for word in tokens:
        if word.isalnum(): # Check if word consists only of alphanumeric characters
            processed_tokens.append(ps.stem(word))

    # 4. Remove stop words
    stopwords_set = set(stopwords.words('english'))
    final_tokens = [word for word in processed_tokens if word not in stopwords_set]

    return " ".join(final_tokens)

# Streamlit App
st.title('Email Spam Classifier')
st.write('Enter an email message to classify it as Spam or Ham.')

user_input = st.text_area('Enter Message Here', height=150)

if st.button('Classify'):
    if user_input:
        # 1. Preprocess the input message
        transformed_message = preprocess_text(user_input)

        # 2. Vectorize the transformed message
        vectorized_message = vectorizer.transform([transformed_message])

        # 3. Make prediction
        prediction = model.predict(vectorized_message)

        # 4. Display result
        if prediction[0] == 1:
            st.error('This is a Spam message!')
        else:
            st.success('This is a Ham message.')
    else:
        st.warning('Please enter a message to classify.')
