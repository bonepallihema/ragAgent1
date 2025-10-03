import streamlit as st
import pandas as pd
import time
import os
import tempfile
from backend import backend  # Import your backend function
import PyPDF2
import pdfplumber

# Page configuration
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2e86ab);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        text-align: center;
    }
    .response-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #fffaf0;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #ffa500;
        margin: 10px 0px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .question-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0px;
        border-left: 5px solid #2196f3;
    }
    .error-box {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0px;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0px;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #2e86ab);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0px;
    }
    .file-info {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file using multiple methods for better accuracy"""
    try:
        # Method 1: Try pdfplumber first (better for text extraction)
        text = ""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                st.success("✓ PDF text extracted using pdfplumber")
                return text.strip()
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}, trying PyPDF2...")
        
        # Method 2: Fallback to PyPDF2
        uploaded_file.seek(0)  # Reset file pointer
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                st.success("✓ PDF text extracted using PyPDF2")
                return text.strip()
        except Exception as e:
            st.error(f"PyPDF2 also failed: {e}")
        
        # If both methods fail
        st.error("❌ Could not extract text from PDF. The file might be scanned or corrupted.")
        return None
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Extract text from PDF and save as text file
            text_content = extract_text_from_pdf(uploaded_file)
            if text_content is None:
                return None
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_file:
                tmp_file.write(text_content)
                return tmp_file.name
                
        elif file_extension == 'txt':
            # For text files, save directly
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as tmp_file:
                content = uploaded_file.getvalue().decode('utf-8')
                tmp_file.write(content)
                return tmp_file.name
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False
if 'temp_file_path' not in st.session_state:
    st.session_state.temp_file_path = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h1>🤖 SmartDoc AI</h1>
        <p>Intelligent Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📁 Document Upload")
    
    # File upload with enhanced UI
    uploaded_file = st.file_uploader(
        "Choose a document", 
        type=['txt', 'pdf'],
        help="Upload your document for analysis (TXT or PDF)",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Show file info
        file_extension = uploaded_file.name.split('.')[-1].upper()
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        st.markdown(f"""
        <div class="file-info">
            <strong>📄 File Info:</strong><br>
            Name: {uploaded_file.name}<br>
            Type: {file_extension}<br>
            Size: {file_size_mb:.2f} MB
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Load Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                if file_extension.lower() == 'pdf':
                    st.info("📖 Extracting text from PDF...")
                
                temp_path = save_uploaded_file(uploaded_file)
                if temp_path:
                    st.session_state.temp_file_path = temp_path
                    st.session_state.document_loaded = True
                    
                    # Read and analyze the file
                    try:
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        st.session_state.file_info = {
                            'name': uploaded_file.name,
                            'type': file_extension,
                            'size_mb': file_size_mb,
                            'char_count': len(content),
                            'word_count': len(content.split()),
                            'line_count': len(content.splitlines())
                        }
                        
                        st.success(f"✅ **{uploaded_file.name}** loaded successfully!")
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>📊 Document Analysis:</strong><br>
                            • Words: {len(content.split()):,}<br>
                            • Characters: {len(content):,}<br>
                            • Lines: {len(content.splitlines()):,}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing document: {e}")
                else:
                    st.error("❌ Failed to process document")
    
    if st.session_state.document_loaded and st.session_state.temp_file_path:
        st.markdown("---")
        st.subheader("⚙️ AI Settings")
        
        k_value = st.slider(
            "Number of context chunks", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="How many text chunks to use for answering questions"
        )
        
        st.subheader("📈 Performance")
        if st.session_state.chat_history:
            st.metric("Total Questions", len(st.session_state.chat_history))
            if st.session_state.chat_history:
                successful_chats = [chat for chat in st.session_state.chat_history if not chat.get('error') and "Error:" not in chat['answer']]
                success_rate = len(successful_chats) / len(st.session_state.chat_history) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
        
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🗑️ Unload Document", use_container_width=True):
            # Clean up temporary file
            if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                os.unlink(st.session_state.temp_file_path)
            st.session_state.document_loaded = False
            st.session_state.temp_file_path = None
            st.session_state.chat_history = []
            st.session_state.file_info = {}
            st.rerun()

# Main content area
st.markdown('<div class="main-header">🤖 SmartDoc AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask Intelligent Questions About Your Documents</div>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 Analytics Dashboard", "ℹ️ User Guide"])

with tab1:
    if not st.session_state.document_loaded:
        st.markdown("""
        <div class="upload-box">
            <h3>📄 Welcome to SmartDoc AI!</h3>
            <p>Upload your document to start asking intelligent questions.</p>
            <p><strong>Supported formats:</strong></p>
            <p>📝 <strong>TXT</strong> - Text documents</p>
            <p>📖 <strong>PDF</strong> - Portable Document Format</p>
            <p><em>Maximum file size: 10MB</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample document structure
        with st.expander("📋 Supported Document Types"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **📝 Text Files (.txt)**
                - Plain text documents
                - UTF-8 encoding recommended
                - No formatting preserved
                - Fastest processing
                """)
            with col2:
                st.markdown("""
                **📖 PDF Files (.pdf)**
                - Text-based PDFs only
                - Scanned PDFs may not work
                - Formatting is removed
                - Text extraction required
                """)
    else:
        # Chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("💭 Ask a Question")
            
            # Show current document info
            if st.session_state.file_info:
                info = st.session_state.file_info
                st.markdown(f"""
                <div class="file-info">
                    <strong>Current Document:</strong> {info['name']} ({info['type']})<br>
                    <strong>Statistics:</strong> {info['word_count']:,} words, {info['char_count']:,} characters
                </div>
                """, unsafe_allow_html=True)
            
            # Question input
            question = st.text_area(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?\nWhat are the key findings?\nExplain the methodology used...",
                height=100,
                key="question_input"
            )
            
            # Quick question examples
            st.markdown("**💡 Quick Questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Main Topic", use_container_width=True):
                    question = "What is the main topic of this document?"
            with col2:
                if st.button("Key Points", use_container_width=True):
                    question = "What are the key points discussed?"
            with col3:
                if st.button("Summary", use_container_width=True):
                    question = "Provide a brief summary of the document."
        
        with col2:
            st.subheader("🚀 Actions")
            if st.button("Get Answer", type="primary", use_container_width=True, disabled=st.session_state.processing):
                if question and st.session_state.temp_file_path:
                    st.session_state.processing = True
                    
                    # Process the question
                    with st.spinner("🔍 Analyzing document..."):
                        try:
                            # Call your backend function with the correct file path
                            answer, context_chunks = backend(
                                file_path=st.session_state.temp_file_path,
                                query=question,
                                k=k_value
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": answer,
                                "contexts": context_chunks if context_chunks else [],
                                "timestamp": time.time(),
                                "error": "Error:" in answer if answer else True
                            })
                            
                        except Exception as e:
                            error_msg = f"Error processing request: {str(e)}"
                            st.error(f"❌ {error_msg}")
                            
                            # Add error to chat history
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": error_msg,
                                "contexts": [],
                                "timestamp": time.time(),
                                "error": True
                            })
                        
                    st.session_state.processing = False
                    st.rerun()
                elif not question:
                    st.warning("Please enter a question first.")
                else:
                    st.error("No document loaded. Please upload a document first.")
        
        # Display processing status
        if st.session_state.processing:
            st.info("🎯 Processing your question... This may take a few moments.")
        
        # Display chat history
        st.markdown("---")
        st.subheader("📝 Conversation History")
        
        if not st.session_state.chat_history:
            st.info("💬 Your questions and answers will appear here. Ask a question to get started!")
        else:
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # Question
                    st.markdown(f'<div class="question-box"><strong>❓ Question:</strong> {chat["question"]}</div>', unsafe_allow_html=True)
                    
                    # Answer - check if it's an error
                    if chat.get("error") or "Error:" in chat["answer"]:
                        st.markdown(f'<div class="error-box"><strong>❌ Error:</strong> {chat["answer"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="response-box">
                            <strong>🤖 AI Answer:</strong><br>
                            {chat["answer"]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Context sources with expander
                    if chat.get("contexts") and len(chat["contexts"]) > 0:
                        with st.expander(f"🔍 View Source Materials ({len(chat['contexts'])} chunks)"):
                            for j, context in enumerate(chat['contexts']):
                                st.markdown(f"**Source {j+1}:**")
                                st.markdown(f'<div class="context-box">{context}</div>', unsafe_allow_html=True)
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")

with tab2:
    st.subheader("📊 Analytics Dashboard")
    
    if st.session_state.document_loaded and st.session_state.temp_file_path:
        try:
            # Read document stats
            with open(st.session_state.temp_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_questions = len(st.session_state.chat_history)
                st.markdown(f"""
                <div class="metric-box">
                    <h3>📝</h3>
                    <h2>{total_questions}</h2>
                    <p>Questions Asked</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                word_count = len(content.split())
                st.markdown(f"""
                <div class="metric-box">
                    <h3>📄</h3>
                    <h2>{word_count}</h2>
                    <p>Word Count</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                successful_chats = len([chat for chat in st.session_state.chat_history if not chat.get('error') and "Error:" not in chat['answer']])
                total_chats = len(st.session_state.chat_history)
                success_rate = (successful_chats / total_chats * 100) if total_chats > 0 else 0
                st.markdown(f"""
                <div class="metric-box">
                    <h3>✅</h3>
                    <h2>{success_rate:.1f}%</h2>
                    <p>Success Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>⚙️</h3>
                    <h2>{k_value}</h2>
                    <p>Chunk Size</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Document preview
            st.subheader("📋 Document Preview")
            st.text_area("First 500 characters of your document:", value=content[:500] + "..." if len(content) > 500 else content, height=150, disabled=True)
            
            # Question history table
            if st.session_state.chat_history:
                st.subheader("📋 Question History")
                history_data = []
                for i, chat in enumerate(st.session_state.chat_history):
                    status = "✅ Success" if not chat.get('error') and "Error:" not in chat['answer'] else "❌ Error"
                    history_data.append({
                        "Question": chat['question'],
                        "Answer Preview": chat['answer'][:80] + "..." if len(chat['answer']) > 80 else chat['answer'],
                        "Sources Used": len(chat.get('contexts', [])),
                        "Status": status
                    })
                
                df = pd.DataFrame(history_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Export option
                if st.button("📤 Export Conversation History"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="smartdoc_conversation_history.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"Error analyzing document: {e}")
    else:
        st.info("📄 Upload a document to see analytics and insights.")

with tab3:
    st.subheader("🎯 User Guide & Troubleshooting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload Document**: Click 'Choose a document' in sidebar
        2. **Load Document**: Click 'Load Document' button
        3. **Ask Questions**: Type your question and click 'Get Answer'
        4. **Review Results**: Check answers and source materials
        
        ### 📁 Supported Formats
        **Text Files (.txt)**
        - Plain text format
        - UTF-8 encoding recommended
        - Fast processing
        
        **PDF Files (.pdf)**
        - Text-based PDFs only
        - Automatic text extraction
        - Scanned PDFs may not work
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Troubleshooting
        
        **❌ PDF Text Extraction Failed:**
        - Ensure PDF is text-based (not scanned)
        - Try converting to text file
        - Check PDF is not password protected
        
        **❌ 'Unable to load file' Error:**
        - Check file format (.txt or .pdf)
        - Ensure file is not corrupted
        - Try a smaller file first
        
        **❌ No answers generated:**
        - Check your OpenRouter API key
        - Ensure you have API credits
        - Verify document contains relevant content
        """)
    
    st.markdown("---")
    st.subheader("📞 Support")
    st.markdown("""
    If you continue experiencing issues:
    1. **For PDF files**: Ensure they are text-based, not scanned images
    2. **For large files**: Try with smaller documents first
    3. **API issues**: Verify your OpenRouter API key and credits
    4. **Format issues**: Convert documents to plain text (.txt) for best results
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🚀 Powered by RAG Technology | SmartDoc AI v2.0 | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)

# Cleanup temporary files on app close
import atexit
def cleanup():
    if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
        try:
            os.unlink(st.session_state.temp_file_path)
        except:
            pass

atexit.register(cleanup)