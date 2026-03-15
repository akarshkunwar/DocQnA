import streamlit as st
import os
import tempfile
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()
# THE FIX: We use the OpenAI chat wrapper instead!
from langchain_openai import ChatOpenAI
import uuid

st.set_page_config(page_title="DocQnA", page_icon="📄")

# Create a unique ID for this user's session to prevent data contamination
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("📄 DocQnA: Free AI Document Summarizer")
st.write("Upload a PDF to save it to the database, then ask questions about it!")

# --- 1. Load API Keys from Environment ---
hf_api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME", "doc-summary-hf")

if not hf_api_key or not pinecone_api_key:
    st.error("⚠️ API keys not found! Please set HF_TOKEN and PINECONE_API_KEY in your server environment.")
    st.stop()

os.environ["HF_TOKEN"] = hf_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# --- 2. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if st.button("Process & Save to Pinecone"):
    if not uploaded_file:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_documents(docs)
                st.write(f"Created {len(chunks)} document chunks.")

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                pc = Pinecone(api_key=pinecone_api_key)

                if index_name not in pc.list_indexes().names():
                    st.info("Creating new Pinecone index (this takes about 60 seconds)...")
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )

                # --- NEW FIX: Clear old documents in this user's namespace before uploading new one ---
                index = pc.Index(index_name)
                try:
                    index.delete(delete_all=True, namespace=st.session_state.session_id)
                except Exception:
                    # Ignore if the namespace doesn't exist or is already empty
                    pass

                PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name,
                                                   namespace=st.session_state.session_id)
                st.success("✅ Document processed and saved to Pinecone successfully!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                os.remove(tmp_file_path)

st.divider()

# --- 3. Chat Interface ---
st.subheader("Document QnA")
user_query = st.text_input("Ask a question about your uploaded document:")

if st.button("Ask"):
    if not user_query:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            try:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "namespace": st.session_state.session_id})

                # THE FIX: Point the standard OpenAI wrapper EXACTLY at Hugging Face's root v1 router
                llm = ChatOpenAI(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    base_url="https://router.huggingface.co/v1",
                    api_key=hf_api_key,
                    max_tokens=512,
                    temperature=0.7,
                )

                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Use three sentences maximum and keep the answer concise,"
                    "unless you are asked to provide a detailed answer."
                    "Do not exceed 300 words in any situation.\n\n"
                    "{context}"
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                response = rag_chain.invoke({"input": user_query})

                st.markdown("### Answer:")
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888888;
        text-align: center;
        font-size: 0.8rem;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Secured with Cloudflare Origin CA • <b>Now over HTTPS</b>
    </div>
    """,
    unsafe_allow_html=True
)