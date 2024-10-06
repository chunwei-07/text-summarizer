# Import necessary libraries
import streamlit as st
import openai
import jieba  # For Chinese word segmentation
import re
from io import BytesIO
from PyPDF2 import PdfReader
from fpdf import FPDF

openai.api_key = st.secrets["mykey"]

# Function to generate a bullet point summary
def generate_bullet_summary(input_text, language):
    prompt = (
        f"Summarize the text below into concise bullet points in {language}:\n\n"
        f"Text: {input_text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates bullet point summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        bullet_summary = response.choices[0].message['content'].strip()
        return bullet_summary
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate a paragraph summary
def generate_paragraph_summary(input_text, language, length):
    length_prompt = "in a short summary" if length == "Short" else "in a long summary"
    prompt = (
        f"Summarize the text below {length_prompt} in {language}:\n\n"
        f"Text: {input_text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates paragraph summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        paragraph_summary = response.choices[0].message['content'].strip()
        return paragraph_summary
    except Exception as e:
        return f"Error: {str(e)}"

# Function to generate questions and answers based on input text
def generate_questions_and_answers(input_text, language):
    prompt = (
        f"Analyze the input text below and generate 5 essential questions that capture the main points. "
        f"Then, answer each question in detail in {language}:\n\n"
        f"Text: {input_text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates questions and answers based on the input text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        q_and_a = response.choices[0].message['content'].strip()
        return q_and_a
    except Exception as e:
        return f"Error: {str(e)}"

# Function to check if text contains primarily Chinese characters
def is_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

# Word count function for Chinese and other languages
def count_words(text):
    if is_chinese(text):
        words = jieba.lcut(text)
    else:
        words = text.split()
    return len(words)

# Function to generate a PDF file from the summary (supports Chinese)
def generate_pdf(summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("SimSun", "", "fonts/SimSun.ttf", uni=True)
    pdf.set_font("SimSun", size=12)
    pdf.multi_cell(200, 10, txt=summary, align='L')
    pdf_output = BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_content)
    pdf_output.seek(0)
    return pdf_output

# Extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to handle PDF Q&A
def ask_question(pdf_content, question):
    prompt = (
        f"Answer the following question based on the PDF content only, in the same language it is asked. "
        f"If the answer cannot be found in the PDF, respond with 'Information not found from PDF file':\n\n"
        f"Question: {question}\n\n"
        f"PDF content: {pdf_content}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions related to the provided text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        answer = response.choices[0].message['content'].strip()
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Multilingual Text Summarizer with PDF Support")

# Sidebar for language selection
st.sidebar.header("Summarization Settings")
input_languages = st.sidebar.multiselect(
    "Select the summarized language",
    options=["English", "Malay", "Chinese"],
    default=["English"]
)

# Summarization type selection
summary_format = st.sidebar.radio(
    "Choose summarization format",
    ("Bullet points", "Paragraph", "Q&A")
)

# Additional settings for paragraph summarization
if summary_format == "Paragraph":
    length = st.sidebar.radio(
        "Choose summary length",
        ("Short", "Long")
    )

st.sidebar.info("**Note:** The settings above are only for summarization function and do not apply on PDF Q&A.")

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

option = None

# Main area for text input or PDF handling
if pdf_file:
    extracted_text = extract_text_from_pdf(pdf_file)
    pdf_content = extracted_text

    # User selects summarization or Q&A on PDF
    option = st.radio("What would you like to do?", ("Summarize PDF", "Q&A on PDF"))

    if option == "Summarize PDF":
        st.write("### Extracted Text from PDF:")
        input_text = st.text_area(
            "You can review or edit the extracted text before summarizing:",
            value=extracted_text,
            height=200,
            key="input_text_pdf"
        )
        word_count = count_words(input_text)
        st.write(f"Word count: {word_count}/1500 words")
    else:
        question = st.text_input("Ask a question about the PDF:")

else:
    st.write("## Enter your text below:")
    input_text = st.text_area(
        "Write your paragraph here:",
        height=200,
        key="input_text"
    )
    word_count = count_words(input_text)
    st.write(f"Word count: {word_count}/1500 words")

# Display appropriate button based on user selection
if option == "Summarize PDF" or not pdf_file:
    if word_count > 1500:
        st.warning("Your input exceeds the 1500-word limit. Please reduce your word count.")
    else:
        if st.button("Summarize"):
            for language in input_languages:
                st.write(f"### Summarization in {language}:")

                if summary_format == "Bullet points":
                    with st.spinner("Generating bullet points summary..."):
                        summary = generate_bullet_summary(input_text, language)
                elif summary_format == "Paragraph":
                    with st.spinner("Generating paragraph summary..."):
                        summary = generate_paragraph_summary(input_text, language, length)
                else:
                    with st.spinner("Generating Q&A summary..."):
                        summary = generate_questions_and_answers(input_text, language)

                st.write(summary)

                # Provide download buttons for text and PDF
                st.download_button(
                    label="Download Summary as Text File",
                    data=summary,
                    file_name=f"Summary_{language}.txt",
                    mime="text/plain"
                )
                pdf_output = generate_pdf(summary)
                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_output.getvalue(),
                    file_name=f"Summary_{language}.pdf",
                    mime="application/pdf"
                )
else:
    if st.button("Answer Question"):
        if question:
            answer = ask_question(pdf_content, question)
            st.write(f"### Answer: {answer}")
