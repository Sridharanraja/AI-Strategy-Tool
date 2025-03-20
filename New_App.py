import streamlit as st
import os
from docx import Document
from groq import Groq
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import pycountry


# Initialize Streamlit app
st.set_page_config(page_title="AI Strategy Planning Tool", page_icon="\U0001F9E0")

# Function to go to the next step
def next_step():
    st.session_state.step += 1
    st.rerun()

# Function to go to the previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()



API = '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'

# Initialize Groq Client
llm = Groq(api_key=API)

models = {
    "Llama 3 (8B)": (llm, "llama3-8b-8192")}


def generate_response(prompt):
    client, model_id = models["Llama 3 (8B)"]  # Select the correct model
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=2500
    )
    return response.choices[0].message.content  # Extract the response text


# Load and Process Documents (adapt path to your environment)
DATA_DIR = "./sheets/DATA/"   # Update this to your folder containing docx/odt/doc files
VECTOR_STORE_PATH = "vector_store"

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx"))]
    documents = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents.append({"source": os.path.basename(file), "content": content})
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return documents

# Vectorization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

def load_documents():
    """Load existing documents."""
    return []

def load_documents():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".pdf", ".docx"))]
    documents = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            documents.append({"source": os.path.basename(file), "content": content})
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return documents

def process_docx(file_path):
    """Process DOCX file and update vector store."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    text_chunks = [(file_path, chunk) for chunk in text_splitter.split_text(documents[0].page_content)]

    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts([chunk[1] for chunk in text_chunks], metadatas=[{"source": chunk[0]} for chunk in text_chunks])
    else:
        vector_store = FAISS.from_texts(
            [chunk[1] for chunk in text_chunks], 
            embeddings, 
            metadatas=[{"source": chunk[0]} for chunk in text_chunks]
        )

    vector_store.save_local(VECTOR_STORE_PATH)


if os.path.exists(VECTOR_STORE_PATH):
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        vector_store = None
else:
    st.warning("Vector store not found. Creating a new one...")
    vector_store = None


def retrieve_relevant_docs(query):
    results = vector_store.similarity_search_with_score(query, k=3)  # Adjust k as needed

    if not results:  # No relevant docs found
        return None, None

    doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
    doc_texts = "\n".join([
        f"**Source: {res[0].metadata['source']}**\n{res[0].page_content[:1000]}"  # Truncate text
        for res in results if res[0] and res[0].metadata
    ])

    return doc_names, doc_texts  # Return doc names & text

def main():
    st.header("Understanding Your Context")
    st.markdown(":blue[Let’s get started! To serve you best, I need to thoroughly understand your context]") #:red[this is red!]
    # st.header("Q1. What is your organization’s name?")
    # Q1: Organization Name
    # Define global font size for text areas
    text_area_size = 14  # Change this to adjust font size

    # Apply CSS for all text areas
    st.markdown(
        f"""
        <style>
        textarea {{
            font-size: {text_area_size}px !important;
            font-weight: bold !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Q1: Organization Name
    organization_name = st.text_input("**Q1. What is your organization’s name?**")

    # Q2: Country Selection
    countries = [country.name for country in pycountry.countries]
    country = st.selectbox("**Q2. Which country is your organization based in?**", countries)
    # country_list = ["USA", "Canada", "UK", "Germany", "France", "India", "China", "Japan", "Australia"]
    # country = st.selectbox("**Q2. Which country is your organization based in?**", country_list)

    # Q3a: Annual Revenue
    annual_revenue = st.text_input("**Q3a. What is your organization’s annual revenue in USD?**")

    # Q3b: Number of Employees
    num_employees = st.text_input("**Q3b. What is the number of employees in your organization?**")

    # Q4a: Focus Area
    focus_area = st.text_input("**Q4a. To help identify relevant AI use cases, which part of your organization do you want to focus on?**")

    # Q4b: Industry Group (from CSV)
    industries_df = pd.read_csv("industries.csv")
    industry_groups = industries_df["industry_group"].unique().tolist()
    industry_group = st.selectbox("**Q4b. Which industry group best describes your business model?**", industry_groups)

    # Q4c: Industry
    industry = st.text_input("**Q4c. Which industry best describes your business model?**")

    # Q4d: Sub-Industry
    sub_industry = st.text_input("**Q4d. Which sub-industry best describes your business model?**")

    # Q4e: Main Products/Services
    products_services = st.text_area("**Q4e. What are the main products or services?**")

    # Q4f: Key Customers
    key_customers = st.text_area("**Q4f. Who are the key customers?**")

    # Q5: Current AI Usage
    st.markdown("### **Q5. Do you currently use AI in any part of your organization? If so, could you briefly describe these use cases?**", unsafe_allow_html=True)
    ai_use_case_1 = st.text_area("**Q5a. Current AI use case 1**")
    ai_use_case_2 = st.text_area("**Q5b. Current AI use case 2**")
    ai_use_case_3 = st.text_area("**Q5c. Current AI use case 3**")

    # Q6: Business Challenges/Opportunities
    business_challenges = st.text_area("**Q6. Last question before we begin: do you have any particular business challenges or opportunities you want to see addressed in the portion of your organization we are focusing on?**")

    if st.button("Execute"):
        user_input = f"""
        - Organization Name: {organization_name}
        - Country: {country}
        - Annual Revenue in USD: {annual_revenue}
        - Number of Employees: {num_employees}

        **Business Details:**
        - Focus Area: {focus_area}
        - Industry Group: {industry_group}
        - Industry: {industry}
        - Sub-Industry: {sub_industry}
        - Main Products/Services: {products_services}
        - Key Customers: {key_customers}

        **Current AI Usage:**
        - AI Use Case 1: {ai_use_case_1}
        - AI Use Case 2: {ai_use_case_2}
        - AI Use Case 3: {ai_use_case_3}

        **Business Challenges & Opportunities:**
        {business_challenges}
    """


        relevant_docs = retrieve_relevant_docs(user_input)
 
        if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
            context = relevant_docs[1]  # Get document text
        else:
            # data_source = f"**Data Source: {model_name}**"
            context = "No relevant documents found. Using AI model only."
        st.write("Here is the summary of the responses I have got from you:")
        
        full_prompt = f"""
        Below is the survey response more detailed summary from the user:
        Context:\n{context}
        **Organization Details:**
        - Organization Name: {organization_name}
        - Country: {country}
        - Annual Revenue in USD: {annual_revenue}
        - Number of Employees: {num_employees}

        **Business Details:**
        - Focus Area: {focus_area}
        - Industry Group: {industry_group}
        - Industry: {industry}
        - Sub-Industry: {sub_industry}
        - Main Products/Services: {products_services}
        - Key Customers: {key_customers}

        **Current AI Usage:**
        - AI Use Case 1: {ai_use_case_1}
        - AI Use Case 2: {ai_use_case_2}
        - AI Use Case 3: {ai_use_case_3}

        **Business Challenges & Opportunities:**
        {business_challenges}

        Based on this information, please provide detail summary.
        """

        with st.spinner("Generating AI response..."):  # Show loading indicator
            bot_reply = generate_response(full_prompt)
            bot_reply


        # st.session_state.ai_governance = generate_response(full_prompt)

if __name__ == "__main__":
    main()
