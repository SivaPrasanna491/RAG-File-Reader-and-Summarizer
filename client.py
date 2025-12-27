import streamlit as st
import requests
import os

# Configure page
st.set_page_config(
    page_title="RAG File Reader",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for dark mode
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Upload box styling */
    .uploadedFile {
        background-color: #0f3460 !important;
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #533483;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #0f3460;
        color: #e94560;
        border: 2px solid #533483;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #533483 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4);
    }
    
    /* Headers */
    h1 {
        color: #e94560;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        color: #ffffff;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #0f3460;
        color: #4ecca3;
        border-left: 4px solid #4ecca3;
    }
    
    .stError {
        background-color: #0f3460;
        color: #e94560;
        border-left: 4px solid #e94560;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background-color: #0f3460;
        border: 2px dashed #533483;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Response box */
    .response-box {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        border: 2px solid #e94560;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
    }
    
    .response-text {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# API endpoints
UPLOAD_URL = "http://localhost:8000/upload"
QUERY_URL = "http://localhost:8000/query/invoke"

# --- PERSISTENCE LOGIC ---
# Initialize session state from Query Params (URL)
# This allows the state to survive a page reload
if "page" in st.query_params and st.query_params["page"] == "query":
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = True

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'response' not in st.session_state:
    st.session_state.response = None

# Title
st.markdown("<h1>📄 RAG File Reader & Summarizer</h1>", unsafe_allow_html=True)

# Page 1: File Upload
if not st.session_state.file_uploaded:
    st.markdown("### 📤 Upload Your Document")
    st.markdown("Upload a PDF, TXT, or Excel file to get started")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'xlsx', 'xls'],
        help="Supported formats: PDF, TXT, Excel"
    )
    
    if uploaded_file is not None:
        st.info(f"📁 Selected file: **{uploaded_file.name}**")
        
        if st.button("🚀 Upload & Process"):
            with st.spinner("⏳ Processing your file... Please wait"):
                try:
                    # Send file to backend
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(UPLOAD_URL, files=files)
                    
                    if response.status_code == 200:
                        st.session_state.file_uploaded = True
                        # Update URL to persist state
                        st.query_params["page"] = "query"
                        st.success("✅ File uploaded successfully! You can now ask questions.")
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Page 2: Query Interface
else:
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Reset button
    if st.button("🔄 Upload New File"):
        st.session_state.file_uploaded = False
        st.session_state.response = None
        # Clear URL params
        st.query_params.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        help="Ask any question about the uploaded document"
    )
    
    if st.button("🔍 Get Answer"):
        if query.strip() == "":
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("🤔 Thinking... Generating answer"):
                try:
                    # Send query to backend
                    payload = {"input": {"query": query}}
                    response = requests.post(QUERY_URL, json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # LangServe returns output in 'output' field
                        answer = result.get('output', 'No answer received')
                        st.session_state.response = answer
                    else:
                        st.error(f"❌ Query failed: {response.text}")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display response
    if st.session_state.response:
        st.markdown("### 📋 Answer:")
        st.markdown(f"""
        <div class="response-box">
            <div class="response-text">
                {st.session_state.response}
            </div>
        </div>
        """, unsafe_allow_html=True)
