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
import json


# Initialize Streamlit app
st.set_page_config(page_title="AI Strategy Planning Tool", page_icon="\U0001F9E0")


# --- Initialize Session State ---
if "data" not in st.session_state:
    # Create a JSON structure to store all steps
    st.session_state["data"] = {
        "step0": {},
        "step1": {},
        "step2": {},
        "step3": {},
        "step4": {},
        "step5": {}
    }



# Initialize session state
if "step" not in st.session_state:
    # st.session_state.step = 1
    st.session_state["step"] = 1


def go_to_step(step):
    """Function to update the step in session state"""
    st.session_state.step = step

def next_step():
    st.session_state["step"] = min(st.session_state["step"] + 1, 6)

def prev_step():
    st.session_state["step"] = max(st.session_state["step"] - 1, 1)

def navigation_buttons(last_step=False):
    """Display Previous and Next buttons for navigation"""
    col1, col2 = st.columns(2)
    col1.button("←", on_click=prev_step)
    if not last_step:
        col2.button("→", on_click=next_step)



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
DATA_DIR = "./sheets/DATA/NEW/"   # Update this to your folder containing docx/odt/doc files
VECTOR_STORE_PATH = "NEW/vector_store"

# Initialize Embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Load and Process Documents
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

# Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)


# Process DOCX and Create Vector Store
def process_docx(file_path):
    """Process DOCX file and update vector store."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    
    if not documents:
        print("No content extracted from DOCX.")
        return

    text_chunks = [(file_path, chunk) for chunk in text_splitter.split_text(documents[0].page_content)]

    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts([chunk[1] for chunk in text_chunks], metadatas=[{"source": chunk[0]} for chunk in text_chunks])
    else:
        print("Creating a new FAISS vector store...")
        vector_store = FAISS.from_texts(
            [chunk[1] for chunk in text_chunks], 
            embeddings, 
            metadatas=[{"source": chunk[0]} for chunk in text_chunks]
        )

    vector_store.save_local(VECTOR_STORE_PATH)
    print("✅ Vector store saved successfully!")

# Ensure Vector Store is Created
if os.path.exists(VECTOR_STORE_PATH):
    # with st.status("🔄 Loading vector store..."):
    try:
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        # st.success("✅ Vector store loaded successfully!")
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        vector_store = None
else:
    # with st.status("🚀 Creating a new vector store..."):
    # st.warning("Vector store not found. Creating a new one...")
    documents = load_documents()
    
    if not documents:
        st.error("No documents found to initialize the vector store!")
    else:
        text_chunks = [(doc["source"], chunk) for doc in documents for chunk in text_splitter.split_text(doc["content"])]

        if text_chunks:
            vector_store = FAISS.from_texts(
                [chunk[1] for chunk in text_chunks], 
                embeddings, 
                metadatas=[{"source": chunk[0]} for chunk in text_chunks]
            )
            vector_store.save_local(VECTOR_STORE_PATH)
            # st.success("✅ New vector store created successfully!")
        else:
            st.error("No text chunks available to create the vector store!")
print("🎉 Completed setup!")


# Retrieve Relevant Docs
def retrieve_relevant_docs(query):
    if not vector_store:
        return None, None
    
    results = vector_store.similarity_search_with_score(query, k=3)  # Adjust k as needed

    if not results:  # No relevant docs found
        return None, None

    doc_names = list(set(res[0].metadata["source"] for res in results if res[0] and res[0].metadata))
    doc_texts = "\n".join([
        f"**Source: {res[0].metadata['source']}**\n{res[0].page_content[:1000]}"  # Truncate text
        for res in results if res[0] and res[0].metadata
    ])

    return doc_names, doc_texts  # Return doc names & text



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

def step0():
    st.header("Step 0: Understanding Your Context")
    st.markdown(":blue[Let’s get started! To serve you best, I need to thoroughly understand your context]") #:red[this is red!]
    # st.header("Q1. What is your organization’s name?")
    # Q1: Organization Name
    # Define global font size for text areas
    text_area_size = 14  # Change this to adjust font size

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step0"]

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
    organization_name = st.text_input("**Q1. What is your organization’s name?**", 
                                      value=data.get("organization_name", ""))

    # Q2: Country Selection
    countries = [country.name for country in pycountry.countries]
    country = st.selectbox("**Q2. Which country is your organization based in?**", 
                           countries, 
                           index=countries.index(data.get("country", "United States")) if "country" in data else 0)
    # country_list = ["USA", "Canada", "UK", "Germany", "France", "India", "China", "Japan", "Australia"]
    # country = st.selectbox("**Q2. Which country is your organization based in?**", country_list)

    # Q3a: Annual Revenue
    annual_revenue = st.text_input("**Q3a. What is your organization’s annual revenue in USD?**", 
                                   value=data.get("annual_revenue", ""))

    # Q3b: Number of Employees
    num_employees = st.text_input("**Q3b. What is the number of employees in your organization?**", 
                                  value=data.get("num_employees", ""))

    # Q4a: Focus Area
    focus_area = st.text_input("**Q4a. To help identify relevant AI use cases, which part of your organization do you want to focus on?**", 
                               value=data.get("focus_area", ""))
        

    # Q4b: Industry Group (from CSV)
    industries_df = pd.read_csv("industries.csv")
    # sector = industries_df["sector"].unique().tolist()
    sector = st.selectbox("**Q4b. Which industry sector best describes your business model?**",industries_df["sector"].unique().tolist(),index=0)

    
    # Q4c & Q4d pre-process
    filtered_df = industries_df[industries_df["sector"] == sector]
    industry_groups = filtered_df["industry_group"].unique().tolist()
    sub_industries = filtered_df["sub_industry"].unique().tolist()

    # Q4c: Industry
    industry_groups = st.selectbox("**Q4c. Which industry best describes your business model?**",industry_groups)

    # Q4d: Sub-Industry
    sub_industry = st.selectbox("**Q4d. Which sub-industry best describes your business model?**", 
                                sub_industries, 
                                index=0)

    # Q4e: Main Products/Services
    products_services = st.text_area("**Q4e. What are the main products or services?**", 
                                     value=data.get("products_services", ""))

    # Q4f: Key Customers
    key_customers = st.text_area("**Q4f. Who are the key customers?**", 
                                 value=data.get("key_customers", ""))

    # Q5: Current AI Usage
    st.markdown("### **Q5. Do you currently use AI in any part of your organization? If so, could you briefly describe these use cases?**", unsafe_allow_html=True)
    ai_use_case_1 = st.text_area("**Q5a. Current AI use case 1**", 
                                 value=data.get("ai_use_case_1", ""))
    ai_use_case_2 = st.text_area("**Q5b. Current AI use case 2**", 
                                 value=data.get("ai_use_case_2", ""))
    ai_use_case_3 = st.text_area("**Q5c. Current AI use case 3**", 
                                 value=data.get("ai_use_case_3", ""))

    # Q6: Business Challenges/Opportunities
    business_challenges = st.text_area("**Q6. Last question before we begin: do you have any particular business challenges or opportunities you want to see addressed in the portion of your organization we are focusing on?**", 
                                       value=data.get("business_challenges", ""))

    if "step0" not in st.session_state:
        st.session_state.it_assessment = None

    st.text_area("**Previous AI Response:**", 
             value=data.get("bot_reply", ""), 
             height=100, 
             disabled=True)

    if st.button("Execute"):        
        full_prompt = f"""
        Below is the survey response more detailed summary from the user:
        **Organization Details:**
        - Organization Name: {organization_name}
        - Country: {country}
        - Annual Revenue in USD: {annual_revenue}
        - Number of Employees: {num_employees}

        **Business Details:**
        - Focus Area: {focus_area}
        - Sector: {sector}
        - Industry Group: {industry_groups}
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
                    # Store response in session state for step1
        st.session_state["step0"] = bot_reply  
        st.session_state["data"]["step0"]["bot_reply"] = bot_reply
        st.session_state["data"]["step0"] = {
            "organization_name": organization_name,
            "country": country,
            "annual_revenue": annual_revenue,
            "num_employees": num_employees,
            "focus_area": focus_area,
            "sector": sector,
            "industry_group": industry_groups,
            "sub_industry": sub_industry,
            "products_services": products_services,
            "key_customers": key_customers,
            "ai_use_case_1": ai_use_case_1,
            "ai_use_case_2": ai_use_case_2,
            "ai_use_case_3": ai_use_case_3,
            "business_challenges": business_challenges,
            "bot_reply": st.session_state["data"]["step0"].get("bot_reply", "")
        }


    # navigation_buttons()
    st.button("Next", on_click=next_step)
        # st.session_state.ai_governance = generate_response(full_prompt)

def step1():
    # navigation_buttons()
    navigation_buttons(last_step=True)
    st.header("Step 1: Construct Value Chains")
        # Ensure step0 output exists before proceeding
    if "step0" not in st.session_state:
        st.warning("Please complete Step 0 first before proceeding.")
        return

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step1"]
    
    st.text_area("**Previous AI Response:**", 
         value=data.get("step1_output", ""), 
         height=100, 
         disabled=True)
    
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(st.session_state["step0"])

            if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
                context = relevant_docs[1]  # Get document text
            else:
                context = "No relevant documents found. Using AI model only."        
            # st.write("Here is the summary of the responses I have got from you:")

            full_prompt = f"""
            Context: {context}
            Construct value chains for the problem statement and business challenges identified in previous step and list primary and support activities {st.session_state.step0}
            """
            with st.spinner("Generating AI response..."):
                step1_output = generate_response(full_prompt)
                step1_output

            # Store step1 output
            st.session_state["step1"] = step1_output
            st.session_state["data"]["step1"]["step1_output"] = step1_output
            st.session_state["data"]["step1"] = {
            "step1_output": st.session_state["data"]["step1"].get("step1_output", "")
            }

            # st.write("### AI Output for Step 1:")
            # st.write(step1_output)

    # navigation_buttons()
    st.button("Next", on_click=next_step)
def step2():
    # navigation_buttons()
    navigation_buttons(last_step=True)
    st.header("Step 2: AI Use Case Identification")
    if "step1" not in st.session_state:
        st.warning("Please complete Step 1 first before proceeding.")
        return
    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step2"]

    st.text_area("**Previous AI Response:**", 
     value=data.get("step2_output", ""), 
     height=200, 
     disabled=True)
    
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(st.session_state["step1"])

            if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
                context = relevant_docs[1]  # Get document text
            else:
                context = "No relevant documents found. Using AI model only."        
            # st.write("Here is the summary of the responses I have got from you:")

            full_prompt = f"""
            Context: {context}
            Convert these value chains into AI use cases and provide the list {st.session_state.step1}
            """
            with st.spinner("Generating AI response..."):
                step2_output = generate_response(full_prompt)
                step2_output

            # Store step1 output
            st.session_state["step2"] = step2_output
            st.session_state["data"]["step2"]["step2_output"] = step2_output
            st.session_state["data"]["step2"] = {
            "step2_output": st.session_state["data"]["step2"].get("step2_output", "")
            }


            # st.write("### AI Output for Step 1:")
            # st.write(step2_output)
    # navigation_buttons()
    st.button("Next", on_click=next_step)

def step3():
    # navigation_buttons()
    navigation_buttons(last_step=True)
    st.header("Step 3: AI Use case prioritization based on Effort-Impact Matrix")
    if "step2" not in st.session_state:
        st.warning("Please complete Step 2 first before proceeding.")
        return

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step3"]
        
    st.text_area("**Previous AI Response:**", 
     value=data.get("step3_output", ""), 
     height=100, 
     disabled=True)
    
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(st.session_state["step2"])

            if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
                context = relevant_docs[1]  # Get document text
            else:
                context = "No relevant documents found. Using AI model only."        
            # st.write("Here is the summary of the responses I have got from you:")

            full_prompt = f"""
            Context: {context}
            Prioritize the AI use cases  into 4 buckets based on the principles of the effort-impact matrix: {st.session_state.step2}
            """
            with st.spinner("Generating AI response..."):
                step3_output = generate_response(full_prompt)
                step3_output

            # Store step3 output
            st.session_state["step3"] = step3_output
            st.session_state["data"]["step3"]["step3_output"] = step3_output
            st.session_state["data"]["step3"] = {
            "step3_output": st.session_state["data"]["step3"].get("step3_output", "")
            }

            # st.write("### AI Output for Step 1:")
            # st.write(step3_output)
    # navigation_buttons()
    st.button("Next", on_click=next_step)

def step4():
    # navigation_buttons()
    navigation_buttons(last_step=True)
    st.header("Step 4: Develop AI Strategy")
    if "step3" not in st.session_state:
        st.warning("Please complete Step 3 first before proceeding.")
        return
        
    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step4"]
    
    st.text_area("**Previous AI Response:**", 
         value=data.get("step4_output", ""), 
         height=100, 
         disabled=True)
    
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(st.session_state["step3"])

            if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
                context = relevant_docs[1]  # Get document text
            else:
                context = "No relevant documents found. Using AI model only."        
            # st.write("Here is the summary of the responses I have got from you:")

            full_prompt = f"""
            Context: {context}
            For the use cases in the buckets Quick Wins and Strategic Projects, develop a detailed AI strategy and action plan {st.session_state.step3}
            """
            with st.spinner("Generating AI response..."):
                step4_output = generate_response(full_prompt)
                step4_output

            # Store step1 output
            st.session_state["step4"] = step4_output
            st.session_state["data"]["step4"]["step4_output"] = step4_output
            st.session_state["data"]["step4"] = {
            "bot_reply": st.session_state["data"]["step4"].get("step4_output", "")
            }

            # st.write("### AI Output for Step 1:")
            # st.write(step4_output)
    # navigation_buttons()
    st.button("Next", on_click=next_step)

def step5():
    navigation_buttons(last_step=True)
    st.header("Step 5: AI Implementation Plan")
    if "step4" not in st.session_state:
        st.warning("Please complete Step 4 first before proceeding.")
        return

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step5"]

    st.text_area("**Previous AI Response:**", 
         value=data.get("step5_output", ""), 
         height=100, 
         disabled=True)
    
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            relevant_docs = retrieve_relevant_docs(st.session_state["step4"])

            if relevant_docs and relevant_docs[0] and relevant_docs[1]:           
                context = relevant_docs[1]  # Get document text
            else:
                context = "No relevant documents found. Using AI model only."        
            # st.write("Here is the summary of the responses I have got from you:")

            full_prompt = f"""
            Context: {context}
            For the AI strategy, create a detailed implementation plan. Please have details on the following topics as part of the implementation plan.            
            1.	Assess AI skills
            2.	Acquire AI skills
            3.	Access AI resources
            4.	Prioritize AI use cases
            5.	Create an AI proof of concept
            6.	Implement responsible AI
            7.	Estimate delivery timelines 
            {st.session_state.step4}
            """
            with st.spinner("Generating AI response..."):
                step5_output = generate_response(full_prompt)
                step5_output

            # Store step1 output
            st.session_state["step5"] = step5_output
            st.session_state["data"]["step5"]["step5_output"] = step5_output
            st.session_state["data"]["step5"] = {
            "bot_reply": st.session_state["data"]["step5"].get("step5_output", "")
            }

            # st.write("### AI Output for Step 1:")
            # st.write(step5_output)

    # navigation_buttons(last_step=True)


if st.session_state.step == 1:
    step0()
elif st.session_state.step == 2:
    step1()
elif st.session_state.step == 3:
    step2()
elif st.session_state.step == 4:
    step3()
elif st.session_state.step == 5:
    step4()
elif st.session_state.step == 6:
    step5()
