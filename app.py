__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import sqlite3
import json
import uuid
import os
from docx import Document
from groq import Groq
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from crewai import Agent, Task
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf():
    """Generate a PDF with collected AI strategy information"""
    pdf_filename = "AI_Strategy_Report.pdf"
    
    # Create PDF
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    y_position = 750  # Initial Y position

    c.drawString(200, y_position, "AI Strategy Report")
    c.line(50, y_position - 10, 550, y_position - 10)  # Underline
    y_position -= 30  # Move cursor down

    # Retrieve stored information
    sections = {
        "Industry": st.session_state.get("industry", "Not Provided"),
        "AI Talent Development Plan": st.session_state.get("ai_talent_plan", "Not Generated"),
        "AI Governance and Ethics": st.session_state.get("ai_governance", "Not Generated"),
        "Investment in AI Infrastructure and Tools": st.session_state.get("ai_investment", "Not Generated"),
    }

    for title, content in sections.items():
        c.drawString(50, y_position, f"📌 {title}:")
        y_position -= 20
        text_lines = content.split("\n")  # Handle multi-line text
        for line in text_lines:
            c.drawString(70, y_position, line)
            y_position -= 15  # Move down for each line

        y_position -= 20  # Add space before next section

    c.save()
    
    return pdf_filename  # Return the filename for download


def download_pdf_button():
    """Button to download the generated PDF"""
    if st.button("📥 Download AI Strategy Report as PDF"):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Click to Download PDF",
                data=file,
                file_name="AI_Strategy_Report.pdf",
                mime="application/pdf",
            )



# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1

# Function to go to the next step
def next_step():
    st.session_state.step += 1
    st.rerun()

# Function to go to the previous step
def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1
        st.rerun()


# Database Connection
conn = sqlite3.connect("chat_database.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    chat_id TEXT PRIMARY KEY,
    user_id TEXT,
    chat_name TEXT,
    messages TEXT,
    model TEXT,
    agent TEXT
)
""")
conn.commit()
try:
    cursor.execute("ALTER TABLE chats ADD COLUMN agent TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass  # Column already exists
API = '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'

# Initialize Groq Client
llm = Groq(api_key=API)

models = {
    "Llama 3 (8B)": (llm, "llama3-8b-8192")}

# AI-Powered Response Generation
# def generate_response(prompt):
#     return llm(prompt)

def generate_response(prompt):
    client, model_id = models["Llama 3 (8B)"]  # Select the correct model
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1500
    )
    return response.choices[0].message.content  # Extract the response text


# Load and Process Documents (adapt path to your environment)
DATA_DIR = "D:/Everse Ai/Groq/DATA/DATA/DATA/"   # Update this to your folder containing docx/odt/doc files
VECTOR_STORE_PATH = "D:/Everse Ai/Groq/code/vector_store"

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

def gather_company_info():
    """Step 1: Gather Company Information"""
    st.header("1. Gather Company Information")

    company_name = st.text_input("Company Name")
    # Ensure industry type is stored in session state
    if "industry" not in st.session_state:
        st.session_state["industry"] = "Manufacturing"  # Default selection

    industry = st.selectbox("Industry", ["Manufacturing", "Hi-tech", "Pharma", "Financial"])
    revenue = st.selectbox("Company Revenue", ["Less than 10M", "10-100M", "100M-1B", "1B plus"])
    resources = st.selectbox("Resource Information", ["10-20 people", "20-100 people", "100-1000 people", "1000+ people"])
    st.session_state["industry"] = industry 

    if st.button("Execute"):
        query = f"Company: {company_name}, Industry: {industry}, Revenue: {revenue}, Resources: {resources}"
        retrieved_docs = retrieve_relevant_docs(query)
        response = generate_response(f"Based on the company's details: {query}, here are the insights: {retrieved_docs}")
        st.write(response)

    st.button("Next", on_click=next_step)

def define_business_objectives():
    """Step 2: Define Business Objectives"""
    st.header("2. Define Business Objectives")

    business_goals = st.text_area("Identify key business goals and challenges:")
    # business_objectives = st.text_area("Define Business Objectives:")
        # Retrieve stored industry from session state
    industry = st.session_state.get("industry", "Pharma")  # Default to Pharma if not set
    
    industry_subheaders = {
            "Manufacturing": [
                "Definition", "Process Automation", "Quality Control",
                "Production Planning", "Supply Chain Management",
                "Workforce Safety", "Sustainability Initiatives"
            ],
            "Hi-tech": [
                "Definition", "AI and Machine Learning", "Cybersecurity",
                "Cloud Computing", "Innovation Strategies", "Data Analytics",
                "Product Development"
            ],
            "Pharma": [
                "Definition", "Drug Discovery Process", "Clinical Trials",
                "Manufacturing Process Optimization", "Supply Chain Management",
                "Marketing and Sales", "Patient Care Management"
            ],
            "Financial": [
                "Definition", "Risk Management", "Fraud Detection",
                "Automated Trading", "Regulatory Compliance", "Customer Insights",
                "Investment Strategies"
            ]
        }


    if st.button("Execute"):
        # Get the corresponding subheaders for the selected industry
        subheaders = industry_subheaders.get(industry, industry_subheaders["Manufacturing"])  # Default to Pharma

        prompt = f"""
        Based on the {industry} industry, generate one **short** insight for each of the following:
        
        1. {subheaders[0]}
        2. {subheaders[1]}
        3. {subheaders[2]}
        4. {subheaders[3]}
        5. {subheaders[4]}
        6. {subheaders[5]}
        7. {subheaders[6]}

        Please ensure each response starts with the corresponding number (e.g., "1.", "2.", etc.).
        """

        response = generate_response(prompt)  # Call LLM function

        # Split only by numbered responses
        insights = {}
        for line in response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():  # Check if it starts with a number
                parts = line.split(".", 1)  # Split only on first period
                if len(parts) > 1:
                    insights[int(parts[0])] = parts[1].strip()

        # **Display insights correctly**
        st.subheader(f"Business Insights for {industry} Industry")
        for idx, subheader in enumerate(subheaders, start=1):
            st.subheader(f"{idx}. {subheader}")
            st.write(insights.get(idx, "No response generated."))  # Avoid missing values

    navigation_buttons()

def assess_current_state():
    """Step 3: Assess Current State"""
    st.header("3. Assess Current State")

    # Initialize session state for storing results
    if "it_assessment" not in st.session_state:
        st.session_state.it_assessment = None
    if "data_assessment" not in st.session_state:
        st.session_state.data_assessment = None
    if "ai_assessment" not in st.session_state:
        st.session_state.ai_assessment = None

    # === IT Infrastructure & Applications ===
    it_infra = st.selectbox("IT Infrastructure", ["On-Prem", "Cloud", "Hybrid"])
    it_apps = st.text_area("IT Applications")

    if st.button("Execute IT Assessment"):
        full_prompt = f"Assess the IT Infrastructure ({it_infra}) and the given IT Applications: {it_apps}."
        st.session_state.it_assessment = generate_response(full_prompt)  # Save result in session state

    if st.session_state.it_assessment:
        st.subheader("🔹 IT Assessment Results")
        st.write(st.session_state.it_assessment)

    st.divider()

    # === Data Infrastructure ===
    data_maturity = st.selectbox("Data Infrastructure Maturity", ["High", "Medium", "Low"])
    data_sources = st.text_area("Important Data Sources")

    if st.button("Execute Data Assessment"):
        full_prompt = f"Assess the data maturity level ({data_maturity}) and analyze these data sources: {data_sources}."
        st.session_state.data_assessment = generate_response(full_prompt)

    if st.session_state.data_assessment:
        st.subheader("🔹 Data Assessment Results")
        st.write(st.session_state.data_assessment)

    st.divider()

    # === AI Skills & Talent ===
    ai_skills = st.text_area("AI Skills and Talent Available")

    if st.button("Execute AI Talent Assessment"):
        full_prompt = f"Evaluate the AI skills and talent available: {ai_skills}."
        st.session_state.ai_assessment = generate_response(full_prompt)

    if st.session_state.ai_assessment:
        st.subheader("🔹 AI Talent Assessment Results")
        st.write(st.session_state.ai_assessment)


def recommend_value_chains():
    """Step 4: Recommend Value Chains"""
    st.header("4. Recommend Value Chains")

    # Ensure previous step data is available
    it_assessment = st.session_state.get("it_assessment", "No IT Assessment available.")
    data_assessment = st.session_state.get("data_assessment", "No Data Assessment available.")
    ai_assessment = st.session_state.get("ai_assessment", "No AI Talent Assessment available.")

    if st.button("Execute"):
        full_prompt = f"""
        Based on the following assessments, recommend value chains:
        - IT Assessment: {it_assessment}
        - Data Assessment: {data_assessment}
        - AI Talent Assessment: {ai_assessment}

        Provide structured value chains based on the inputs.
        """
        st.session_state.value_chains = generate_response(full_prompt)  # Call LLM

    if "value_chains" in st.session_state:
        st.subheader("🔹 Recommended Value Chains")
        st.write(st.session_state.value_chains)

    navigation_buttons()

def identify_ai_opportunities():
    """Step 5: Identify AI Opportunities and Use Cases"""
    st.header("5. Identify AI Opportunities and Use Cases")

    # Get industry type from previous step (Step 1)
    selected_industry = st.session_state.get("industry", "General")

    if st.button("Execute AI Opportunities"):
        full_prompt = f"""
        Based on the industry type: {selected_industry}, identify AI opportunities and use cases.
        Provide 7 structured sub-headers, with a brief one-line description for each.
        """

        # Call LLM
        st.session_state.ai_opportunities = generate_response(full_prompt)

    # Display AI Opportunities if available
    if "ai_opportunities" in st.session_state:
        st.subheader("🔹 Identified AI Opportunities")
        st.write(st.session_state.ai_opportunities)

    navigation_buttons()

def develop_ai_strategy():
    """Step 6: Develop AI Strategy, Vision, and Roadmap"""
    st.header("6. Develop AI Strategy, Vision, and Roadmap")

    # Get industry type from previous step
    selected_industry = st.session_state.get("industry", "General")

    # AI Vision Statement
    st.subheader("🔹 AI Vision Statement")
    if st.button("Execute AI Vision Statement"):
        full_prompt = f"Create a concise AI Vision Statement for a company in the {selected_industry} industry."
        st.session_state.ai_vision_statement = generate_response(full_prompt)

    if "ai_vision_statement" in st.session_state:
        st.success(st.session_state.ai_vision_statement)

    # AI Roadmap
    st.subheader("🔹 Establish an AI Roadmap")
    if st.button("Execute AI Roadmap Established"):
        full_prompt = f"Define a step-by-step AI Roadmap for a company in the {selected_industry} industry."
        st.session_state.ai_roadmap = generate_response(full_prompt)

    if "ai_roadmap" in st.session_state:
        st.success(st.session_state.ai_roadmap)

    # Prioritize AI Initiatives
    st.subheader("🔹 Prioritize AI Initiatives")
    if st.button("Execute Prioritize AI Initiatives"):
        full_prompt = f"List the top AI initiatives to prioritize in the {selected_industry} industry."
        st.session_state.ai_initiatives = generate_response(full_prompt)

    if "ai_initiatives" in st.session_state:
        st.success(st.session_state.ai_initiatives)

    navigation_buttons()



def develop_ai_execution_plan():
    """Step 7: Develop AI Execution Plan"""
    st.header("7. Develop AI Execution Plan")

    # Get industry type from previous steps
    selected_industry = st.session_state.get("industry", "General")

    # Detailed Action Plan
    st.subheader("🔹 Detailed Action Plan")
    if st.button("Execute Action Plan"):
        full_prompt = f"Create a detailed AI action plan for a company in the {selected_industry} industry."
        st.session_state.ai_action_plan = generate_response(full_prompt)

    if "ai_action_plan" in st.session_state:
        st.success(st.session_state.ai_action_plan)

    # AI Metrics and Benchmarks
    st.subheader("🔹 AI Metrics and Benchmarks")
    if st.button("Execute AI Metrics and Benchmarks"):
        full_prompt = f"Define key AI metrics and benchmarks for a company in the {selected_industry} industry."
        st.session_state.ai_metrics = generate_response(full_prompt)

    if "ai_metrics" in st.session_state:
        st.success(st.session_state.ai_metrics)

    # Prioritize AI Initiatives
    st.subheader("🔹 Prioritize AI Initiatives")
    if st.button("Execute Prioritize AI Initiatives"):
        full_prompt = f"List and prioritize AI initiatives for a company in the {selected_industry} industry."
        st.session_state.ai_priorities = generate_response(full_prompt)

    if "ai_priorities" in st.session_state:
        st.success(st.session_state.ai_priorities)

    # AI Strategy Refinement Process
    st.subheader("🔹 AI Strategy Refinement Process")
    if st.button("Execute AI Strategy Refinement Process"):
        full_prompt = f"Describe an AI strategy refinement process for a company in the {selected_industry} industry."
        st.session_state.ai_refinement = generate_response(full_prompt)

    if "ai_refinement" in st.session_state:
        st.success(st.session_state.ai_refinement)

    navigation_buttons()


def build_manage_ai_capabilities():
    """Step 8: Build and Manage AI Capabilities"""
    st.header("8. Build and Manage AI Capabilities")

    # Get industry type from previous steps
    selected_industry = st.session_state.get("industry", "General")

    # AI Talent Development Plan
    st.subheader("🔹 AI Talent Development Plan")
    if st.button("Execute AI Talent Development Plan"):
        full_prompt = f"Develop an AI talent development plan for a company in the {selected_industry} industry."
        st.session_state.ai_talent_plan = generate_response(full_prompt)

    if "ai_talent_plan" in st.session_state:
        st.success(st.session_state.ai_talent_plan)

    # AI Governance and Ethics
    st.subheader("🔹 AI Governance and Ethics")
    if st.button("Execute AI Governance and Ethics"):
        full_prompt = f"Describe AI governance and ethical considerations for a company in the {selected_industry} industry."
        st.session_state.ai_governance = generate_response(full_prompt)

    if "ai_governance" in st.session_state:
        st.success(st.session_state.ai_governance)

    # Investment in AI Infrastructure and Tools
    st.subheader("🔹 Investment in AI Infrastructure and Tools")
    if st.button("Execute Investment Plan"):
        full_prompt = f"Suggest AI infrastructure and tools investment strategies for a company in the {selected_industry} industry."
        st.session_state.ai_investment = generate_response(full_prompt)

    if "ai_investment" in st.session_state:
        st.success(st.session_state.ai_investment)

    download_pdf_button()


    navigation_buttons(last_step=True)


def navigation_buttons(last_step=False):
    """Display Previous and Next buttons for navigation"""
    col1, col2 = st.columns(2)
    col1.button("Previous", on_click=prev_step)
    if not last_step:
        col2.button("Next", on_click=next_step)

# Step Controller
st.title("AI Strategy Planning Tool")

if st.session_state.step == 1:
    gather_company_info()
elif st.session_state.step == 2:
    define_business_objectives()
elif st.session_state.step == 3:
    assess_current_state()
elif st.session_state.step == 4:
    recommend_value_chains()
elif st.session_state.step == 5:
    identify_ai_opportunities()
elif st.session_state.step == 6:
    develop_ai_strategy()
elif st.session_state.step == 7:
    develop_ai_execution_plan()
elif st.session_state.step == 8:
    build_manage_ai_capabilities()
