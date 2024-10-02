# Import necessary libraries
import streamlit as st
import openai
import jieba   # For Chinese word segmentation
import re
from io import BytesIO
from PyPDF2 import PdfReader
from fpdf import FPDF

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
    if is_chinese(text):  # If the text contains Chinese characters
        words = jieba.lcut(text)
    else:                 # Default to splitting by spaces for non-Chinese languages
        words = text.split()
    return len(words)

# Function to generate a PDF file from the summary (supports Chinese)
def generate_pdf(summary):
    pdf = FPDF()
    pdf.add_page()

    # Add a font that supports Chinese characters (e.g., SimSun or Noto Sans CJK)
    # You can download and use a font like "SimSun" (SimSun.ttf) or "Noto Sans CJK"
    pdf.add_font("SimSun", "", "fonts/SimSun.ttf", uni=True)  # Add the TTF font file
    pdf.set_font("SimSun", size=12)  # Use the font with Chinese support

    # Write the summary to the PDF
    pdf.multi_cell(200, 10, txt=summary, align='L')

    # Save the PDF to a BytesIO stream
    pdf_output = BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin1')  # Use appropriate encoding
    pdf_output.write(pdf_content)
    pdf_output.seek(0)  # Move the cursor to the start of the stream
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
    prompt = f"Answer the following question based on the PDF content only. If the question is unrelated to the PDF, respond with 'There is no related information found in the PDF file.': {question}.\n\nPDF content: {pdf_content}"

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
st.sidebar.header("Summarisation Settings")
input_languages = st.sidebar.multiselect(
    "Select the summarized language",
    options=["English", "Malay", "Chinese"],
    default=["English"]
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

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize 'option' with a default value
option = None

# Main area for text input or PDF handling
if pdf_file:
    extracted_text = extract_text_from_pdf(pdf_file)  # Extracted text from PDF
    pdf_content = extracted_text  # Store extracted content in pdf_content

    # Allow user to select summarization or Q&A
    option = st.radio("What would you like to do?", ("Summarize PDF", "Q&A on PDF"))

    if option == "Summarize PDF":
        st.write("### Extracted Text from PDF:")
        input_text = st.text_area(
            "You can review or edit the extracted text before summarizing:",
            value=extracted_text,
            height=200,
            key="input_text_pdf"
        )
        # Calculate word count and limit to 1500 words for summarization
        word_count = count_words(input_text)
        st.write(f"Word count: {word_count}/1500 words")
    else:
        question = st.text_input("Ask a question about the PDF:")
else:
    # Initialize session state for input text
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    # Set default 'option' to summarization for manual text input (no PDF case)
    option = "Summarize"

    # Main area for manual text input
    st.write("## Enter your text below:")
    input_text = st.text_area(
        "Write your paragraph here:",
        value=st.session_state["input_text"],
        height=200,
        key="input_text"
    )
    # Calculate word count and limit to 1500 words for manual text input
    word_count = count_words(input_text)
    st.write(f"Word count: {word_count}/1500 words")

# Display appropriate button depending on the user's choice
if option == "Summarize PDF" or option == "Summarize":
    if word_count > 1500:
        st.warning("Your input exceeds the 1500-word limit. Please reduce your word count.")
    else:
        if st.button("Summarize"):
            if pdf_file and option == "Summarize PDF":
                for language in input_languages:
                    st.write(f"### Summary in {language}:")
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(input_text, language, summary_type, length if summary_type == "Paragraph" else None)
                    st.write(summary)

                    # Provide download buttons for text and PDF
                    st.download_button(
                        label="Download Summary as Text File",
                        data=summary,
                        file_name=f"summary_{language}.txt",
                        mime="text/plain"
                    )
                    pdf_output = generate_pdf(summary)
                    st.download_button(
                        label="Download Summary as PDF",
                        data=pdf_output.getvalue(),
                        file_name=f"summary_{language}.pdf",
                        mime="application/pdf"
                    )
            elif input_text:
                for language in input_languages:
                    st.write(f"### Summary in {language}:")
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(input_text, language, summary_type, length if summary_type == "Paragraph" else None)
                    st.write(summary)

                    # Provide download buttons for text and PDF
                    st.download_button(
                        label="Download Summary as Text File",
                        data=summary,
                        file_name=f"summary_{language}.txt",
                        mime="text/plain"
                    )
                    pdf_output = generate_pdf(summary)
                    st.download_button(
                        label="Download Summary as PDF",
                        data=pdf_output.getvalue(),
                        file_name=f"summary_{language}.pdf",
                        mime="application/pdf"
                    )
else:
    if st.button("Answer Question"):
        if question:
            answer = ask_question(pdf_content, question)
            st.write(f"### Answer: {answer}")
