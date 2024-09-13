# Import necessary libraries
import streamlit as st
import openai

openai.api_key = st.secrets["mykey"]

# Summarization Function
def summarize_text(input_text, language):
    prompt = f"Summarize this paragraph in {language}: {input_text}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        # The completion for ChatCompletion API is in 'message' -> 'content'
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        return f"Error: {str(e)}"




  # Streamlit UI
st.title("Multilingual Text Summarizer")

# Sidebar for language selection
st.sidebar.header("Settings")
languages = st.sidebar.multiselect(
    "Choose the language(s) for summarization",
    options=["English", "Malay", "Chinese"],
    default=["English"]
)

if not languages:
    st.sidebar.warning("Please select at least one language.")

# Main area for text input
st.write("## Enter your text below: ")

input_text = st.text_area(
    "Write your paragraph here:",
    height=200
)

# Output
if st.button("Summarize"):
    if input_text:
        for language in languages:
            st.write(f"### Summary in {language}:")
            summary = summarize_text(input_text, language)
            st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")
