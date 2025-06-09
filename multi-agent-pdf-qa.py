import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="Your_API_KEY")

# FAISS and embeddings
embedding_dim = 384
faiss_index = faiss.IndexFlatL2(embedding_dim)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

document_store = []

# -------------- BASIC FUNCTIONS --------------

def extract_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text("text")
    return text

def split_text_into_chunks(text, chunk_size=500, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def create_embedding(text):
    embedding = embed_model.encode([text], convert_to_numpy=True)
    return embedding[0]

def index_pdf_and_store_text(pdf_path):
    global document_store
    text = extract_pdf_text(pdf_path)
    chunks = split_text_into_chunks(text)
    document_store = chunks

    for chunk in chunks:
        embedding = create_embedding(chunk)
        faiss_index.add(np.array([embedding]))

# -------------- AGENTS --------------

# Retrieval Agent
def retrieval_agent(user_query, k=5):
    query_embedding = create_embedding(user_query)
    distances, indices = faiss_index.search(np.array([query_embedding]), k)
    retrieved_texts = [document_store[i] for i in indices[0]]

    print(f"\n [Retrieval Agent Output]:\n")
    for idx, text in enumerate(retrieved_texts):
        print(f"Document {idx+1}:\n{text[:300]}...\n")

    return indices[0]

# Summarizer Agent
def summarizer_agent(retrieved_indices):
    context = "\n".join([document_store[i] for i in retrieved_indices])

    summarizer_prompt = f"""You are an academic researcher. Summarize the following context professionally:

Context:
{context}

Summary:"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(summarizer_prompt)
    summary = response.text.strip()

    print(f"\n [Summarizer Agent Output]:\n\n{summary}\n")
    return summary

# Q&A Agent
def qa_agent(summary, user_query):
    qa_prompt = f"""You are a helpful assistant. Based on the following summary, answer the user question clearly and friendly.

Summary:
{summary}

User Question:
{user_query}

Answer:"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(qa_prompt)
    answer = response.text.strip()

    print(f"\n [Q&A Agent Output]:\n\n{answer}\n")
    return answer

# Critique Agent
def critique_agent(answer):
    critique_prompt = f"""You are a strict reviewer. Review and polish this answer professionally.

Answer:
{answer}

Reviewed Answer:"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(critique_prompt)
    reviewed_answer = response.text.strip()

    print(f"\n [Critique Agent Output]:\n\n{reviewed_answer}\n")
    return reviewed_answer

# Router Agent (Dynamic decision maker)
def router_agent(user_query):
    router_prompt = f"""Decide if the following user question requires information from a provided document (e.g., a research paper, manual, book).

User Question:
{user_query}

Respond ONLY with "DOCUMENT" or "GENERAL"."""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(router_prompt)
    decision = response.text.strip().upper()
    print(f"\n [Router Agent Decision]: {decision}\n")
    return decision

# General Answer Agent
def general_answer_agent(user_query):
    prompt = f"""You are an intelligent assistant. Answer the following question directly, since it is not document-related.

Question:
{user_query}

Answer:"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    answer = response.text.strip()

    print(f"\n [General Answer Agent Output]:\n\n{answer}\n")
    return answer

# -------------- FULL PIPELINE --------------

def multi_agent_dynamic_pipeline(user_query):
    decision = router_agent(user_query)

    if decision == "DOCUMENT":
        retrieved_indices = retrieval_agent(user_query)
        summary = summarizer_agent(retrieved_indices)
        raw_answer = qa_agent(summary, user_query)
        final_answer = critique_agent(raw_answer)
    else:
        final_answer = general_answer_agent(user_query)

    print("\n [Final Answer Returned]:\n")
    print(final_answer)

# -------------- USAGE --------------

if __name__ == "__main__":
    pdf_path = "/content/Research paper final project.pdf"  # <-- upload your file
    index_pdf_and_store_text(pdf_path)

    while True:
        user_query = input("\n Ask anything (type 'exit' to stop): ")
        if user_query.lower() == 'exit':
            break
        multi_agent_dynamic_pipeline(user_query)
