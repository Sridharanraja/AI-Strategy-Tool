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
import pdfkit
import tempfile
import io
from reportlab.pdfgen import canvas
import textwrap 
from PyPDF2 import PdfMerger

# Initialize Streamlit app
st.set_page_config(page_title="AI Strategy Planning Tool", page_icon="\U0001F9E0")

torch.classes.__path__ = []

def generate_pdf(step_name, step_data):
    """Generate a properly formatted PDF for each step."""
    pdf_buffer = io.BytesIO()  # Create an in-memory buffer
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    
    # Define fonts
    normal_font = "Helvetica"
    bold_font = "Helvetica-Bold"

    y_position = 750  # Initial Y position
    c.setFont(bold_font, 14)  # Title font
    c.drawString(200, y_position, f"Report: {step_name}")
    c.line(50, y_position - 10, 550, y_position - 10)  # Underline
    y_position -= 30  # Move cursor down

    # Process and format text
    c.setFont(normal_font, 12)  # Reset to normal font
    if step_data:
        for line in step_data.split("\n"):
            # Detect bold text using Markdown format (**bold**)
            if line.startswith("**") and line.endswith("**"):
                c.setFont(bold_font, 12)  # Apply bold font
                line = line.replace("**", "")  # Remove asterisks
            else:
                c.setFont(normal_font, 12)  # Reset to normal font

            wrapped_lines = textwrap.wrap(line, width=90)  # Wrap long text
            for wrapped_line in wrapped_lines:
                c.drawString(50, y_position, wrapped_line)
                y_position -= 15  # Move down for each line

                # Handle page overflow
                if y_position < 50:
                    c.showPage()
                    c.setFont(normal_font, 12)  # Reset font on new page
                    y_position = 750
    else:
        c.drawString(50, y_position, "No Data Available")

    c.save()
    pdf_buffer.seek(0)  # Move buffer position to start
    
    return pdf_buffer  # Return the generated PDF buffer

def download_pdf_button(step_name, data_key, file_name):
# def download_pdf_button(step_name, file_name):
    """Button to download PDF for a specific step."""
    step_data = st.session_state["data"].get(step_name, {}).get(data_key, "")  #get("bot_reply", "")

    if step_data:
        pdf_file = generate_pdf(step_name, step_data)
        st.download_button(
            label=f"📥 Download {file_name}",
            data=pdf_file,
            file_name=f"{file_name}.pdf",
            mime="application/pdf",
        )
    else:
        print("No data available for {file_name}, please execute first.")
        # st.warning(f"No data available for {file_name}, please execute first.")

def download_all_pdfs():
    """Combine all step PDFs into a single PDF and download."""
    merger = PdfMerger()

    # List of step names and corresponding keys
    steps = [
        ("step0", "bot_reply", "Step0_Report"),
        ("step1", "step1_output", "Step1_Report"),
        ("step2", "step2_output", "Step2_Report"),
        ("step3", "step3_output", "Step3_Report"),
        ("step4", "step4_output", "Step4_Report"),
        ("step5", "step5_output", "Step5_Report"),
        ("step6", "step6_output", "Step6_Report")
    ]
    
    for step_name, data_key, file_name in steps:
        step_data = st.session_state["data"].get(step_name, {}).get(data_key, "")

        if step_data:
            # Generate PDF (already a BytesIO object)
            pdf_file = generate_pdf(step_name, step_data)
    
            # Directly append the BytesIO object
            pdf_file.seek(0)
            merger.append(pdf_file)

    # Create a buffer for the merged PDF
    merged_pdf_buffer = io.BytesIO()

    # Write the merged content to the buffer
    merger.write(merged_pdf_buffer)
    merger.close()

    # Reset buffer pointer to the start
    merged_pdf_buffer.seek(0)

    # Button to download the merged PDF
    st.download_button(
        label="📥 Download All Reports",
        data=merged_pdf_buffer.getvalue(),
        file_name="All_Steps_Reports.pdf",
        mime="application/pdf"
    )


# --- Initialize Session State ---
if "data" not in st.session_state:
    # Create a JSON structure to store all steps
    st.session_state["data"] = {
        "step0": {},
        "step1": {},
        "step2": {},
        "step3": {},
        "step4": {},
        "step5": {},
        "step6": {}
    }



# Initialize session state
if "step" not in st.session_state:
    # st.session_state.step = 1
    st.session_state["step"] = 1


def go_to_step(step):
    """Function to update the step in session state"""
    st.session_state.step = step

def next_step():
    st.session_state["step"] = min(st.session_state["step"] + 1, 7)

def prev_step():
    st.session_state["step"] = max(st.session_state["step"] - 1, 1)

def navigation_buttons(last_step=False):
    """Display Previous and Next buttons for navigation"""
    col1, col2 = st.columns(2)
    col1.button("←", on_click=prev_step)
    if not last_step:
        col2.button("→", on_click=next_step)



API = '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' #'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'

# Initialize Groq Client
llm = Groq(api_key=API)

models = {
    "Llama 3 (8B)": (llm, "llama3-8b-8192"),
    "llama-3 (versatile)": (llm, "llama-3.3-70b-versatile"),
    "mixtral-8x7b-32768": (llm, "llama3-8b-8192")
}



# def generate_response(prompt):
#     client, model_id = models["Llama 3 (8B)"]  # Select the correct model
#     response = client.chat.completions.create(
#         model=model_id,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.5,
#         max_tokens=2500
#     )
#     return response.choices[0].message.content  # Extract the response text


def generate_response(prompt):
    client, model_id = models["Llama 3 (8B)"]  # Select the correct model
    
    # Define the AI's role and expertise
    system_message = {
        "role": "system",
        "content": ("""
        You are AI strategy expert from Ximerai with more than 20 years of experience in AI strategy and consulting,
        Technology and Management consulting firms like McKinsey Consulting and Accenture.
        You Your role is to systematically guide users in identifying and implementing the most suitable AI use cases based on their inputs,
        size (revenue and headcount), industry, capabilities, and strategic needs.
        Throughout the interaction, maintain a consultative, expert tone.
        Focus on practical business outcomes rather than technical specifications when explaining the value of each recommended use case.
        """
        )
    }
    
    # User's input
    user_message = {"role": "user", "content": prompt}

    # Generate response
    response = client.chat.completions.create(
        model=model_id,
        messages=[system_message, user_message],
        temperature=0.5,
        max_tokens=5500
    )

    return response.choices[0].message.content


#only for step5 -------------------------------------------
def generate_response5(prompt):
    client, model_id = models["llama-3 (versatile)"]  # Select the correct model
    
    # Define the AI's role and expertise
    system_message = {
        "role": "system",
        "content": ("""
        You are an AI Project management and Implementation expert from Ximerai with strong project and program management abilities
        """
        )
    }
    
    # User's input
    user_message = {"role": "user", "content": prompt}

    # Generate response
    response = client.chat.completions.create(
        model=model_id,
        messages=[system_message, user_message],
        temperature=0.5,
        max_tokens=2500
    )

    return response.choices[0].message.content
#----------------------------------------------------------

#only for step6 -------------------------------------------
def generate_response6(prompt):
    client, model_id = models["Llama 3 (8B)"]  # Select the correct model
    
    # Define the AI's role and expertise
    system_message = {
        "role": "system",
        "content": ("""
        You are a Technology expert from Ximerai and in the past you have worked in companies like Microsoft, AWS, Accenture and Deloitte.
        Please provide detailed technology implementation plan  accessing the companies Data infrastructure,
        Cloud infrastructure, AI infrastructure and Integration infrastructure
        """
        )
    }
    
    # User's input
    user_message = {"role": "user", "content": prompt}

    # Generate response
    response = client.chat.completions.create(
        model=model_id,
        messages=[system_message, user_message],
        temperature=0.5,
        max_tokens=2500
    )

    return response.choices[0].message.content
#----------------------------------------------------------


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

    text_area_size = 2  # Change this to adjust font size

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step0"]

   
    st.markdown(
        """
        <style>
            .custom-font {
                font-size: 2px;  /* Change font size */
                font-weight: bold;  /* Optional: Make it bold */
                
            }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Q1: Organization Name
    organization_name = st.text_input("**Q1. What is the name of the company? Example: Uber, Amazon, Cognizant.**",value=data.get("organization_name",""))
    
    # Q2: Num of Employees and Revenue 
    num_employees_and_revenue = st.text_input("**Q2. Tell us more about the size of your company. Share details about number of employees and revenue.**",value=data.get("num_employees_and_revenue",""))

    # Q3: Industry, sectors and segments
    industry_sectors_segments = st.text_input("**Q3. Give more details about the industry, sectors, segments you operate in. Example: Healthcare, Software, Retail, Beverages, Consulting**",value=data.get("industry_sectors_segments",""))

    # Q4: Country
    country = st.text_input("**Q4. What country are you headquartered? What countries you operate in?**",value=data.get("country",""))

    # Q5: Key Customers
    key_customers = st.text_input("**Q5. What are your main products or services? Who are your key customers? Are you more B2B or B2C type of company?**",value=data.get("key_customers",""))

    # Q6:Competitors
    competitors = st.text_input("**Q6. Who are your competitors? What are some of your strengths against them?**",value=data.get("competitors",""))

    # Q7: AI Partner
    ai_partner = st.text_input("**Q7. What are some of your business objectives, challenges and goals where you see scope for improvement through the use of AI?**",value=data.get("ai_partner",""))

    #Q8: Challenges and Goals
    objectives_challenges_goals = st.text_input("**Q8. How do you rate your company in terms of AI maturity and adoption? What are the top three AI use cases you have implemented?**",value=data.get("objectives_challenges_goals",""))

    #Q9: AI Maturity Adoption
    AI_maturity_adoption = st.text_input("**Q9. Are you engaged with an AI consulting company/partner? If yes, can you please provide the name?**",value=data.get("AI_maturity_adoption",""))


    if "step0" not in st.session_state:
        st.session_state.it_assessment = None

    st.text_area("**Previous AI Response:**", 
             value=data.get("bot_reply", ""), 
             height=100, 
             disabled=True)

    if st.button("Execute"):        
        full_prompt = f"""
        Provide an Executive Summary based on the responses received:
        Organization: {organization_name},Num of Employees and Revenue :{num_employees_and_revenue},
        Industry, sectors and segments:{industry_sectors_segments},Country: {country},
        key Customers: {key_customers},
        Competitors: {competitors}, Competitors:{competitors},
        AI Partner: {ai_partner}, Objectives, Challenges and Goals:{objectives_challenges_goals},
        AI Maturity Adoption and 3 AI use cases: {AI_maturity_adoption}
        Based on the information gathered from the user and information available in the public domain and on the internet, please provide a detailed executive summary. The summary should include topics like Business challenges and Goals, Competitive positioning and landscape, Industry trends and growth prospects, Current AI readiness and maturity and how AI can help to achieve that. 
        """
  

        with st.spinner("Generating AI response..."):  # Show loading indicator
            bot_reply = generate_response(full_prompt)
            bot_reply
        # Store response in session state for step1
        st.session_state["step0"] = bot_reply  
        st.session_state["data"]["step0"]["bot_reply"] = bot_reply
        st.session_state["data"]["step0"] = {
            "organization_name": organization_name,
            "num_employees_and_revenue": num_employees_and_revenue,
            "industry_sectors_segments": industry_sectors_segments,
            "country": country,
            "key_customers": key_customers,
            "competitors": competitors,
            "ai_partner": ai_partner,
            "objectives_challenges_goals": objectives_challenges_goals,
            "AI_maturity_adoption": AI_maturity_adoption,
            "bot_reply": st.session_state["data"]["step0"].get("bot_reply", "")
        }

    download_pdf_button("step0","bot_reply","Step0_Report")
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

            full_prompt = f"""
            Based on the industry, sectors, segments, product and services captured in the user inputs, Construct value chains based on the concept. Also, list primary and support activities. {st.session_state.step0}
            """
            # Construct value chains for the problem statement and business challenges identified in previous step and list primary and support activities. Provide a rational on why these value chains are constructed. {st.session_state.step0}
            
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
    download_pdf_button("step1","step1_output","Step1_Report")
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

            full_prompt = f"""
            Based on the value chains proposed by Ximerai in the previous step and the business goal and objective captured from the user inputs, Suggest top 5-7 AI use cases. Provide justification and business case for selecting these use cases. {st.session_state.step1}
            """
            # Based on the value chains created in the previous step, Suggest top 5 AI use cases for the problem statement and business challenges suggested by the user. Provide detail business justification for selecting these use cases. {st.session_state.step1}
            
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
    download_pdf_button("step2","step2_output","Step2_Report")
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
           
            full_prompt = f"""
            For the Identified use cases in the previous step, Categorize the AI use cases into 4 categories based on the principles of the effort-impact matrix. Provide scoring, detailed rationale and explanation for the categorization. The four categories are high impact-low effort, high impact- high effort, low impact – low effort and low impact – high effort.
            The top 3 AI use cases in the high impact-low effort and high impact- high effort category, Prioritize the ones which are in the high impact-low effort category. For the 3 shortlisted AI use cases, highlight data and AI infrastructure considerations, regulatory and compliance considerations, data privacy and security concerns, technical implementation challenges and ethical implications.
            For the 3 shortlisted use cases, suggest technical and infrastructure needed for each, the gaps between the current and the required infrastructure and risk profiles. Based on the gap analysis and risk assessment, rank the 3 use cases by implementation feasibility.{st.session_state.step2}
            """
            # For the Identified use cases in the previous step, Categorize the AI use cases  into 4 buckets based on the principles of the effort-impact matrix. Provide scoring, detailed rationale and explanation for the categorization. {st.session_state.step2}
            
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
    download_pdf_button("step3","step3_output","Step3_Report")
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
            full_prompt = f"""
            Build a detailed document for “AI Strategy and Implementation action Plan”.  The detailed AI strategy document should cover the following points in detail. Write atleast half a page for each for each of these points:
            1. Executive Summary
            2. Current State Assessment
            2.1 Company Profile
            2.2 Current AI Initiatives
            2.3 AI Landscape in the industry
            3. AI Strategy Framework
            3.1 Vision and Mission
            3.2 Strategic Objectives
            3.3 Strategic Pillars
            3.4 Governance Model
            3.5 Data Strategy
            4. Company-Specific AI Opportunities
            4.1 Opportunity Mapping
            4.2 Prioritization Framework
            4.3 Implementation Considerations
            5. Implementation Roadmap
            5.1 Phased Approach
            5.2 Timeline and Milestones
            5.3 Critical Dependencies
            5.4 Change Management
            5.5 Risk Management
            6. Resource Requirements
            6.1 Talent and Skills
            6.2 Budget
            6.3 Technology Infrastructure
            6.4 Organizational Structure
            7. Conclusion and Next Steps

            The AI action plan should detail time frames, budget requirements and methodology needed for the following topics.          
            1. Assess AI skills
            2. Acquire AI skills
            3. Access AI resources
            4. Prioritize AI use cases
            5. Create an AI proof of concept
            6. Implement responsible AI
            7. Estimate delivery timelines 
            8. Establishment of AI Center of Excellence
            {st.session_state.step3}
            """

            #old : For the use cases identified in the High impact category, develop a detailed AI strategy and implementation the action plan. The AI statergy should include components like Vision and Goals, KPI's and Metrics, Current AI Readyness Assessment of capabilities,  Required AI infrastructure and technologies, AI Roadmap. Include a detailed section on Data and Infrastructure requirement, Talent and Skills requirement, Ethics and Governance, Change Management and Adoption, Continuous Monitoring and Evaluation 
            with st.spinner("Generating AI response..."):
                step4_output = generate_response(full_prompt)
                step4_output
            
            # Store step4 output
            st.session_state["step4"] = step4_output
            st.session_state["data"]["step4"]["step4_output"] = step4_output
            st.session_state["data"]["step4"] = {
            "step4_output": st.session_state["data"]["step4"].get("step4_output", "")
            }

            # st.write("### AI Output for Step 1:")
            # st.write(step4_output)

    download_pdf_button("step4","step4_output","Step4_Report")
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

        full_prompt = f"""
        For the AI strategy in the previous step, create a detailed implementation plan. Please have details on the following topics as part of the implementation plan.  The implementation plan should detail time frames, budget requirements and methodology needed.          
        1. Assess AI skills
        2. Acquire AI skills
        3. Access AI resources
        4. Prioritize AI use cases
        5. Create an AI proof of concept
        6. Implement responsible AI
        7. Estimate delivery timelines 
        8. Establishment of AI Center of Excellence
        {st.session_state.step4} 
        """

        with st.spinner("Generating AI response..."):
            step5_output = generate_response5(full_prompt)
            step5_output

        # Store step5 output
        st.session_state["step5"] = step5_output
        st.session_state["data"]["step5"]["step5_output"] = step5_output
        st.session_state["data"]["step5"] = {
        "step5_output": st.session_state["data"]["step5"].get("step5_output", "")
        }

            # st.write("### AI Output for Step 1:")
            # st.write(step5_output)
    download_pdf_button("step5","step5_output","Step5_Report")
    # navigation_buttons(last_step=True)
    st.button("Next", on_click=next_step)


def step6():
    navigation_buttons(last_step=True)
    st.header("Step 6: AI Technology Implementation Plan")
    if "step5" not in st.session_state:
        st.warning("Please complete Step 5 first before proceeding.")
        return

    # Retrieve existing data or use defaults
    data = st.session_state["data"]["step6"]

    st.text_area("**Previous AI Response:**", 
         value=data.get("step6_output", ""), 
         height=100, 
         disabled=True)
    
    if st.button("Execute"):      

        full_prompt = f"""
        Create a detailed Technology Implementation Project plan which should provide details for components like Data infrastructure,
        Cloud infrastructure, AI infrastructure and Integration infrastructure. Present data in Tabular format and charts.      
        {st.session_state.step5}
        """
        
        with st.spinner("Generating AI response..."):
            step6_output = generate_response6(full_prompt)
            step6_output

        # Store step6 output
        st.session_state["step6"] = step6_output
        st.session_state["data"]["step6"]["step6_output"] = step6_output
        st.session_state["data"]["step6"] = {
        "step6_output": st.session_state["data"]["step6"].get("step6_output", "")
        }

            # st.write("### AI Output for Step 1:")
            # st.write(step5_output)
    download_pdf_button("step6","step6_output","Step6_Report")
    download_all_pdfs()
    # navigation_buttons(last_step=True)
    # st.button("Next", on_click=next_step)


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
elif st.session_state.step == 7:
    step6()
