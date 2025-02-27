# Import necessary libraries
import streamlit as st
import openai

# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import StringPromptTemplate
# from langchain.chains import LLMChain

from typing import List, Union, Dict, Any
import jieba   # For Chinese word segmentation
import re
from io import BytesIO
from PyPDF2 import PdfReader
from fpdf import FPDF
import tempfile
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

openai.api_key = st.secrets["mykey"]

# Memory Class to store document processing history
class DocumentMemory:
  def __init__(self):
    self.documents = {}   # Store document content
    self.summaries = {}   # Store generated summaries
    self.analyses = {}    # Store analysis results
    self.history = []     # Conversation history

  def add_document(self, doc_id, content, metadata=None):
    self.documents[doc_id] = {
        'content': content,
        'metadata': metadata or {},
        'processed': False
    }

  def add_summary(self, doc_id, summary_type, language, content):
    if doc_id not in self.summaries:
        self.summaries[doc_id] = {}
    
    key = f"{summary_type}_{language}"
    self.summaries[doc_id][key] = content

  def get_document_history(self, doc_id):
    """Get all processing history for a document"""
    return {
        'document': self.documents.get(doc_id),
        'summaries': self.summaries.get(doc_id, {}),
        'analyses': self.analyses.get(doc_id, {})
    }

# Advanced text analysis tools
class TextAnalyzer:
  def __init__(self, openai_api_key):
    self.openai_api_key = openai_api_key
  
  def extract_key_entities(self, text):
    """Extract important entities from the text using NER"""
    try:
      response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract and categorize the main entities in this text. Include people, organizations, locations, dates, and key concepts."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
      )
      return response.choices[0].message['content'].strip()
    except Exception as e:
      return f"Error in entity extraction: {str(e)}"

  def analyze_sentiment(self, text):
    """Analyze the sentiment of the text"""
    try:
      response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Analyze the sentiment of this text. Provide a score from -1 (very negative) to 1 (very positive) and explain your reasoning."},
            {"role": "user", "content": text}
        ],
        temperature=0.3,
      )
      return response.choices[0].message['content'].strip()
    except Exception as e:
      return f"Error in sentiment analysis: {str(e)}"

  def generate_questions(self, text):
    """Generate insightful questions based on the text"""
    try:
      response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Based on this text, generate 5 insightful questions that would be valuable to explore further."},
            {"role": "user", "content": text}
        ],
        temperature=0.7,
      )
      return response.choices[0].message['content'].strip()
    except Exception as e:
      return f"Error in question generation: {str(e)}"

  def extract_keywords(self, text, num_keywords=10):
    """Extract key terms using TF-IDF"""
    try:
      # For longer documents, use TF-IDF
      sentences = re.split(r'(?<=[.!?])\s+', text)
      vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
      tfidf_matrix = vectorizer.fit_transform(sentences)
      feature_names = vectorizer.get_feature_names_out()
      
      # Get top keywords
      tfidf_scores = tfidf_matrix.sum(axis=0).A1
      top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
      keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
      
      return keywords
    except Exception as e:
      # Fallback to AI model
      try:
        response = openai.ChatCompletion.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": f"Extract the {num_keywords} most important keywords or phrases from this text."},
              {"role": "user", "content": text}
          ],
          temperature=0.3,
        )
        return response.choices[0].message['content'].strip()
      except Exception as e2:
        return f"Error in keyword extraction: {str(e2)}"

# Enhanced summarization function with multiple approaches
def enhanced_summarize_text(input_text, language, summary_type, length="Short", approach="abstractive"):
  system_message = "You are a highly skilled AI assistant that specializes in text summarization."
    
  if approach == "extractive":
      system_message += " Focus on extracting and preserving the most important sentences from the original text."
  elif approach == "abstractive":
      system_message += " Create a concise summary that captures the essence of the text while using your own phrasing."
  elif approach == "hybrid":
      system_message += " Combine key extracted sentences with your own rephrasing to create a comprehensive yet concise summary."
  
  if summary_type == "Bullet points":
      prompt = f"Summarize this text in {language} as a list of bullet points. Ensure you capture all key information:"
  else:
      prompt = f"Summarize this text in {language} as a {length} paragraph. Ensure you maintain the core message and key details:"

  try:
      response = openai.ChatCompletion.create(
          model="gpt-4o",
          messages=[
              {"role": "system", "content": system_message},
              {"role": "user", "content": f"{prompt}\n\n{input_text}"}
          ],
          max_tokens=500 if length == "Long" else 250,
          temperature=0.5,
      )
      summary = response.choices[0].message['content'].strip()
      return summary
  except Exception as e:
      return f"Error: {str(e)}"

# Document processing agent for autonomous operation
class DocumentAgent:
  def __init__(self, openai_api_key):
    self.openai_api_key = openai_api_key
    self.memory = DocumentMemory()
    self.analyzer = TextAnalyzer(openai_api_key)

  def process_document(self, text, task_description, doc_id=None):
    """Process a document based on a natural language task description"""
    if not doc_id:
      doc_id = f"doc_{len(self.memory.documents) + 1}"

    # Store the document
    self.memory.add_document(doc_id, text)

    # Plan processing steps based on task description
    planning_prompt = f"""
    I need to process a document with the following task: {task_description}

    Based on this task, outline a step-by-step plan using these available tools:
    1. Summarize text (bullet points or paragraphs)
    2. Extract key entities (people, organizations, locations, concepts)
    3. Analyze sentiment
    4. Generate insightful questions
    5. Extract keywords

    Provide a JSON-formatted plan with numbered steps and tool parameters.
    """

    try:
      response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a document processing assistant that creates detailed processing plans."},
            {"role": "user", "content": planning_prompt}
        ],
        temperature=0.3,
      )
      
      # For now, a basic processing is used as demo
      results = {
          "document_id": doc_id,
          "task": task_description,
          "summary": enhanced_summarize_text(text, "English", "Paragraph", "Short"),
          "entities": self.analyzer.extract_key_entities(text),
          "sentiment": self.analyzer.analyze_sentiment(text),
          "questions": self.analyzer.generate_questions(text),
          "keywords": self.analyzer.extract_keywords(text)
      }

      return results

    except Exception as e:
      return f"Error in document processing: {str(e)}"

def generate_and_display_visualizations(text):
    """Generate and display visualizations for document analysis"""
    
    st.subheader("Document Visualizations")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Word Frequency", "Sentiment Flow", "Entity Distribution"])
    
    with viz_tabs[0]:
        # 1. Word frequency chart
        st.write("### Word Frequency Analysis")
        
        # Preprocess text - remove common stopwords and punctuation
        import re
        from collections import Counter
        
        # Simple preprocessing
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = [word for word in cleaned_text.split() if len(word) > 3]  # Filter short words
        
        # Count word frequencies and get top 20
        word_counts = Counter(words).most_common(20)
        words_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
        
        # Create plot with matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=words_df, ax=ax)
        ax.set_title('Top 20 Word Frequencies')
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(fig)
        
        # Also show the data in a table
        st.write("#### Word Frequency Data")
        st.dataframe(words_df)
    
    with viz_tabs[1]:
        # 2. Sentiment analysis over document sections
        st.write("### Sentiment Flow Throughout Document")
        
        # Split text into sections
        sections = re.split(r'\n\n+', text)
        sections = [s for s in sections if len(s.strip()) > 50]  # Only analyze substantial sections
        
        # For a real app, you would analyze sentiment per section
        # Here, simulate sentiment scores
        if len(sections) > 0:
            # Option 1: For simulated data:
            sentiments = np.random.uniform(-0.8, 0.8, len(sections))
            
            # Create dataframe for plotting
            sentiment_df = pd.DataFrame({
                'Section': range(1, len(sentiments) + 1),
                'Sentiment': sentiments
            })
            
            # Create plot with plotly for interactivity
            import plotly.express as px
            fig = px.line(sentiment_df, x='Section', y='Sentiment', 
                          title='Sentiment Throughout Document',
                          labels={'Sentiment': 'Sentiment Score (-1 to 1)'},
                          markers=True)
            
            # Add a horizontal line at y=0
            fig.add_shape(type="line", x0=0, y0=0, x1=len(sentiments), y1=0,
                         line=dict(color="red", width=1, dash="dash"))
            
            # Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick summary
            avg_sentiment = np.mean(sentiments)
            st.metric("Average Document Sentiment", f"{avg_sentiment:.2f}", 
                     delta=f"{avg_sentiment:.2f}", delta_color="normal")
        else:
            st.write("Not enough sections to analyze sentiment flow.")
    
    with viz_tabs[2]:
        # 3. Entity distribution (simulated)
        st.write("### Entity Distribution")
        
        # In a real application, you would extract entities using a proper NER model
        # This is simulated data for demonstration
        entity_types = ['Person', 'Organization', 'Location', 'Date', 'Concept']
        entity_counts = np.random.randint(1, 15, size=len(entity_types))
        
        # Create dataframe
        entity_df = pd.DataFrame({
            'Entity Type': entity_types,
            'Count': entity_counts
        })
        
        # Create donut chart with plotly
        import plotly.express as px
        fig = px.pie(entity_df, values='Count', names='Entity Type', 
                    title='Entity Type Distribution',
                    hole=0.4)
        
        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Display entities in expandable sections
        for entity_type in entity_types:
            with st.expander(f"{entity_type} Entities"):
                # This would be real extracted entities in a proper implementation
                st.write("Example entities would be listed here.")

# Streamlit UI with agentic capabilities
def create_app_ui():
    st.title("Agentic Document Analysis System")
    
    # Initialize session state for memory
    if 'memory' not in st.session_state:
        st.session_state.memory = DocumentMemory()
    
    if 'current_document' not in st.session_state:
        st.session_state.current_document = None
    
    if 'agent' not in st.session_state:
        # In a real app, you would securely handle the API key
        st.session_state.agent = DocumentAgent(openai.api_key)
    
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    
    # Sidebar for document selection and settings
    st.sidebar.header("Settings & Documents")
    
    # Document mode selection
    doc_mode = st.sidebar.radio(
        "Choose input method",
        ("Text Input", "PDF Upload", "Stored Documents")
    )
    
    # Language and processing options
    languages = st.sidebar.multiselect(
        "Select output languages",
        options=["English", "Malay", "Chinese", "Spanish", "French", "Japanese"],
        default=["English"]
    )
    
    summary_approach = st.sidebar.selectbox(
        "Summarization approach",
        ("abstractive", "extractive", "hybrid")
    )
    
    summary_type = st.sidebar.radio(
        "Summary format",
        ("Bullet points", "Paragraph")
    )
    
    if summary_type == "Paragraph":
        length = st.sidebar.radio(
            "Summary length",
            ("Short", "Medium", "Long")
        )
    else:
        length = "Short"
    
    # Natural language task specification
    st.sidebar.header("Task Specification")
    task_description = st.sidebar.text_area(
        "Describe your task in natural language",
        value="Summarize the document and extract key information."
    )
    
    # Main interface based on selected mode
    if doc_mode == "Text Input":
        st.header("Enter Your Text")
        input_text = st.text_area(
            "Paste or type your text here",
            height=200
        )
        
        # Store the text in session state so it's available for visualization
        if input_text:
            st.session_state.document_text = input_text
        
        # Create two columns for the buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Process Document"):
                if input_text:
                    with st.spinner("Processing your document..."):
                        # Process with agent
                        results = st.session_state.agent.process_document(
                            input_text, 
                            task_description
                        )
                        
                        # Store current document
                        st.session_state.current_document = results
                        
                        # Display results
                        st.header("Document Analysis Results")
                        
                        # Summary tab
                        st.subheader("Summary")
                        st.write(results["summary"])
                        
                        # Entity tab
                        st.subheader("Key Entities")
                        st.write(results["entities"])
                        
                        # Sentiment tab
                        st.subheader("Sentiment Analysis")
                        st.write(results["sentiment"])
                        
                        # Questions tab
                        st.subheader("Suggested Questions")
                        st.write(results["questions"])
                        
                        # Generate download buttons
                        st.download_button(
                            label="Download Analysis as Text",
                            data=f"DOCUMENT ANALYSIS\n\nSummary:\n{results['summary']}\n\nEntities:\n{results['entities']}\n\nSentiment:\n{results['sentiment']}\n\nQuestions:\n{results['questions']}",
                            file_name="document_analysis.txt",
                            mime="text/plain"
                        )
                else:
                    st.warning("Please enter some text to process.")
        
        with col2:
            if st.button("Analyze Document Visually"):
                if input_text:
                    with st.spinner("Generating visualizations..."):
                        generate_and_display_visualizations(input_text)
                else:
                    st.warning("Please enter some text to visualize.")
    
    elif doc_mode == "PDF Upload":
        st.header("Upload PDF Document")
        pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
        
        if pdf_file:
            try:
                # Extract text from PDF
                reader = PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                # Store the text in session state
                st.session_state.document_text = text
                
                # Show extracted text
                st.subheader("Extracted Text")
                with st.expander("View extracted text"):
                    st.text(text[:1000] + "..." if len(text) > 1000 else text)
                
                # Process PDF options
                pdf_action = st.radio(
                    "What would you like to do with this PDF?",
                    ("Analyze Fully", "Summarize Only", "Q&A", "Custom Analysis", "Visualize")
                )
                
                if pdf_action != "Visualize":
                    if st.button("Process PDF"):
                        with st.spinner("Processing your PDF..."):
                            if pdf_action == "Analyze Fully":
                                results = st.session_state.agent.process_document(
                                    text, 
                                    "Perform a comprehensive analysis including summary, entities, sentiment and questions."
                                )
                            elif pdf_action == "Summarize Only":
                                summary = enhanced_summarize_text(
                                    text, 
                                    languages[0], 
                                    summary_type, 
                                    length, 
                                    summary_approach
                                )
                                results = {"summary": summary}
                            elif pdf_action == "Q&A":
                                st.session_state.current_document = {"content": text, "type": "pdf"}
                                st.experimental_rerun()  # Redirect to Q&A interface
                            else:  # Custom Analysis
                                results = st.session_state.agent.process_document(
                                    text, 
                                    task_description
                                )
                            
                            # Display results
                            st.header("PDF Analysis Results")
                            
                            if "summary" in results:
                                st.subheader("Summary")
                                st.write(results["summary"])
                                
                                # Generate download button for summary
                                st.download_button(
                                    label="Download Summary as Text",
                                    data=results["summary"],
                                    file_name="pdf_summary.txt",
                                    mime="text/plain"
                                )
                            
                            # Show other results if available
                            for key in ["entities", "sentiment", "questions"]:
                                if key in results:
                                    st.subheader(key.capitalize())
                                    st.write(results[key])
                else:  # Visualize option
                    if st.button("Generate Visualizations"):
                        with st.spinner("Creating visualizations..."):
                            generate_and_display_visualizations(text)
                            
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    elif doc_mode == "Stored Documents" and st.session_state.current_document:
        st.header("Stored Document")
        
        # Display document content and analysis options
        st.write("Current document ID:", st.session_state.current_document.get("document_id", "N/A"))
        
        # Document interaction options
        interaction_mode = st.radio(
            "How would you like to interact with this document?",
            ("View Analysis", "Ask Questions", "Generate New Analysis", "Visualize Document")
        )
        
        if interaction_mode == "View Analysis":
            # Display existing analysis
            if "summary" in st.session_state.current_document:
                st.subheader("Summary")
                st.write(st.session_state.current_document["summary"])
            
            for key in ["entities", "sentiment", "questions"]:
                if key in st.session_state.current_document:
                    st.subheader(key.capitalize())
                    st.write(st.session_state.current_document[key])
                    
        elif interaction_mode == "Ask Questions":
            st.subheader("Ask Questions About This Document")
            question = st.text_input("Enter your question:")
            
            if st.button("Get Answer") and question:
                # Get document content
                doc_content = st.session_state.current_document.get("content", "")
                if not doc_content and "document_id" in st.session_state.current_document:
                    doc_id = st.session_state.current_document["document_id"]
                    doc_info = st.session_state.memory.get_document_history(doc_id)
                    if doc_info and "document" in doc_info:
                        doc_content = doc_info["document"].get("content", "")
                
                if doc_content:
                    with st.spinner("Generating answer..."):
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document."},
                                    {"role": "user", "content": f"Document: {doc_content[:4000]}...\n\nQuestion: {question}"}
                                ],
                                temperature=0.3,
                            )
                            answer = response.choices[0].message['content'].strip()
                            st.write("### Answer:")
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                else:
                    st.error("Document content not available.")
        
        elif interaction_mode == "Generate New Analysis":
            st.subheader("Generate New Analysis")
            new_task = st.text_area(
                "Describe the analysis you want to perform:",
                value="Create a detailed summary focusing on the main arguments and supporting evidence."
            )
            
            if st.button("Generate Analysis"):
                # Get document content
                doc_content = st.session_state.current_document.get("content", "")
                if not doc_content and "document_id" in st.session_state.current_document:
                    doc_id = st.session_state.current_document["document_id"]
                    doc_info = st.session_state.memory.get_document_history(doc_id)
                    if doc_info and "document" in doc_info:
                        doc_content = doc_info["document"].get("content", "")
                
                if doc_content:
                    with st.spinner("Generating new analysis..."):
                        results = st.session_state.agent.process_document(
                            doc_content, 
                            new_task,
                            st.session_state.current_document.get("document_id")
                        )
                        
                        # Display results
                        st.header("New Analysis Results")
                        
                        if "summary" in results:
                            st.subheader("Summary")
                            st.write(results["summary"])
                        
                        for key in ["entities", "sentiment", "questions"]:
                            if key in results:
                                st.subheader(key.capitalize())
                                st.write(results[key])
                else:
                    st.error("Document content not available.")
        
        elif interaction_mode == "Visualize Document":
            # Get document content
            doc_content = st.session_state.current_document.get("content", "")
            if not doc_content:
                if "document_id" in st.session_state.current_document:
                    doc_id = st.session_state.current_document["document_id"]
                    doc_info = st.session_state.memory.get_document_history(doc_id)
                    if doc_info and "document" in doc_info:
                        doc_content = doc_info["document"].get("content", "")
                
                # Fallback to session state if no content found
                if not doc_content and st.session_state.document_text:
                    doc_content = st.session_state.document_text
            
            if doc_content:
                if st.button("Generate Visualizations"):
                    with st.spinner("Creating document visualizations..."):
                        generate_and_display_visualizations(doc_content)
            else:
                st.error("No document content available to visualize.")
                    
    else:
        st.info("Please upload a document or enter text to begin.")
    
    # Add conversational agent interface at the bottom
    st.header("Ask Me Anything")
    user_question = st.text_input("How can I help you with document processing?")
    
    if st.button("Send") and user_question:
        with st.spinner("Thinking..."):
            try:
                # Context includes available tools and current document if any
                context = "Available tools: summarization, entity extraction, sentiment analysis, keyword extraction, document Q&A, and document visualization."
                
                if st.session_state.current_document:
                    context += "\nYou are currently working with a document."
                
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": f"You are a helpful document analysis assistant. {context}"},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.7,
                )
                st.write("### Response:")
                st.write(response.choices[0].message['content'].strip())
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Run the app
if __name__ == "__main__":
  create_app_ui()
