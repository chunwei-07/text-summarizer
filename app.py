# Import necessary libraries
import streamlit as st
import openai

openai.api_key = st.secrets["mykey"]

# Summarization function
def summarize_text(input_text, language):
    prompt = f"Summarize this paragraph in {language}: {input_text}"

    try:
        response = openai.ChatCompletion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        summary = response.choices[0].text.strip()
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
