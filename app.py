# Import necessary libraries
import streamlit as st
import openai
import jieba   # For Chinese word segmentation
import re

openai.api_key = st.secrets["mykey"]

# Summarization Function
def summarize_text(input_text, language, summary_type, length="Short"):
    if summary_type == "Bullet points":
        prompt = f"Summarize this paragraph in {language} in bullet points: {input_text}"
    else:
        prompt = f"Summarize this paragraph in {language} as a {length} paragraph: {input_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150 if length == "Short" else 300,
            temperature=0.7,
        )
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Function to check if text contains primarily Chinese characters
def is_chinese(text):
    # Count Chinese characters using a regular expression
    return bool(re.search('[\u4e00-\u9fff]', text))

# Word count function for Chinese and other languages
def count_words(text):
    if is_chinese(text):  # If the text contains Chinese character
        words = jieba.lcut(text)
    else:                 # Default to splitting by spaces for non-Chinese languages
        words = text.split()
    return len(words)

# Streamlit UI
st.title("Multilingual Text Summarizer")

# Sidebar for language selection
st.sidebar.header("Settings")
input_language = st.sidebar.selectbox(
    "Select the input language",
    options=["English", "Malay", "Chinese"],
    index=0  # Default to English
)

# Summarization type (bullet points or paragraph)
summary_type = st.sidebar.radio(
    "Choose summarization type",
    ("Bullet points", "Paragraph")
)

# Additional settings for paragraph summarization
if summary_type == "Paragraph":
    length = st.sidebar.radio(
        "Choose summary length",
        ("Short", "Long")
    )

# Initialize session state for input text
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Main area for text input
st.write("## Enter your text below:")

# Display text area with session state to track changes
input_text = st.text_area(
    "Write your paragraph here:",
    value=st.session_state["input_text"],
    height=200,
    key="input_text",    # Use key to trigger session state update
)

# Calculate word count for selected languages
word_count = count_words(st.session_state["input_text"])

# Display word count information
st.write(f"Word count: {word_count}/800 words")

# Output
if st.button("Summarize"):
    if word_count > 800:
        st.warning("Your input exceeds the 800-word limit. Please shorten your text.")
    elif input_text:
        st.write(f"### Summary in {input_language}:")
        summary = summarize_text(input_text, input_language, summary_type, length if summary_type == "Paragraph" else None)
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
