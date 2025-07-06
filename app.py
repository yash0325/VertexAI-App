import streamlit as st
import os
import tempfile
import json
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from vertexai import init as vertexai_init
from dotenv import load_dotenv

# -- Vertex AI authentication for Streamlit Community Cloud --
from google.oauth2 import service_account

# Load Google credentials from Streamlit secrets
creds_dict = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

PROJECT_ID = st.secrets["PROJECT_ID"]
REGION = st.secrets["REGION"]

# (Optional: If running locally, you can still load .env)
if os.getenv("PROJECT_ID") is None:
    load_dotenv()

# ---- Streamlit page setup ----
st.set_page_config(page_title="PDF Q&A (Vertex AI Gemini)", layout="wide")
st.markdown(
    "<h1 style='text-align:center;'>📄 PDF Q&A Chatbot with Vertex AI Gemini</h1>"
    "<p style='text-align:center;'>Ask questions about your PDF! Powered by Gemini.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# ---- Helper: Load and split PDF ----
def load_and_split(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = []
    for doc in docs:
        split_docs.extend(splitter.create_documents([doc.page_content]))
    return split_docs

# ---- Main App Logic ----
if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF uploaded successfully!")

    # Build vectorstore with embeddings and Vertex AI auth
    @st.cache_resource(show_spinner=False)
    def build_vectorstore(path):
        vertexai_init(
            project=PROJECT_ID,
            location=REGION,
            credentials=credentials,
        )
        docs = load_and_split(path)
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    vectorstore = build_vectorstore(file_path)

    # Q&A section
    st.header("Ask your PDF a question!")
    user_query = st.text_input("Enter your question about the PDF:")

    if st.button("Get Answer", key="submit_query") and user_query:
        with st.spinner("Thinking..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join([doc.page_content for doc in docs])
            llm = ChatVertexAI(model="gemini-2.5-flash", temperature=0.2)
            prompt = (
                f"Use ONLY the following PDF excerpts to answer the user's question.\n"
                f"Question: {user_query}\n"
                f"PDF Excerpts:\n{context}\n"
                "Answer (short and clear):"
            )
            answer_obj = llm.invoke(prompt)

            # Safely extract only the answer text
            if hasattr(answer_obj, "content"):
                answer_text = answer_obj.content
            elif hasattr(answer_obj, "text"):
                answer_text = answer_obj.text
            elif isinstance(answer_obj, dict) and "content" in answer_obj:
                answer_text = answer_obj["content"]
            else:
                answer_text = str(answer_obj)  # fallback

            st.subheader("Gemini's Answer")
            st.success(answer_text)
            with st.expander("Show Retrieved Excerpts"):
                st.write(context)
else:
    st.info("Please upload a PDF to get started.")

st.markdown("---")
st.caption("Built with Google Vertex AI Gemini and LangChain • POC-ready 🚀")
