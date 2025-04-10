import os
import json
import requests
import streamlit as st
import PyPDF2
import spacy
import numpy as np
import pandas as pd
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import tempfile
import faiss
import uuid
import re

# Set page configuration
st.set_page_config(
    page_title="MedGuide",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    @classmethod
    def validate_token(cls):
        if not cls.HF_TOKEN:
            st.sidebar.error("Hugging Face API Token is required. Please set it as an environment variable.")
            return False
        return True

class RAGDatabase:
    def __init__(self):
        self.claims_data = []
        self.embeddings = None
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_healthfc_data(self, healthfc_data):
        """Load and index health claims data for RAG retrieval"""
        self.claims_data = healthfc_data.get("health_claims", [])
        
        # Extract texts for embedding
        texts = [claim.get("claim", "") + " " + claim.get("explanation", "") 
                for claim in self.claims_data]
        
        # Generate embeddings
        self.embeddings = self.embedder.encode(texts)
        
        # Create FAISS index for fast similarity search
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
        return len(self.claims_data)
    
    def search_claims(self, query, top_k=3):
        """Search for relevant claims using semantic similarity"""
        if not self.claims_data or self.index is None:
            return []
            
        # Embed the query
        query_embedding = self.embedder.encode([query])
        
        # Search the index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k=min(top_k, len(self.claims_data))
        )
        
        # Return the top results with relevance scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.claims_data):
                relevance_score = 1.0 - (distances[0][i] / 100.0)  # Convert distance to similarity
                result = {
                    **self.claims_data[idx],
                    "relevance_score": max(0, min(relevance_score, 1.0))  # Ensure score is between 0-1
                }
                results.append(result)
        
        return results

# Load Datasets
@st.cache_data
def load_datasets():
    datasets = {}
    
    # Load HealthFC Dataset
    healthfc_path = 'healthfc.json'
    if os.path.exists(healthfc_path):
        try:
            with open(healthfc_path, 'r') as f:
                datasets['healthfc'] = json.load(f)
        except Exception as e:
            st.sidebar.warning(f"Could not load health claims database: {e}")
    else:
        # Create a sample structure for demonstration
        datasets['healthfc'] = {
            "health_claims": [
                {
                    "claim": "Garlic may help lower blood sugar levels in diabetes",
                    "evidence_level": "Medium",
                    "explanation": "Some studies suggest that garlic, particularly aged garlic extract, may help reduce fasting blood sugar and improve insulin sensitivity in people with type 2 diabetes. The active compounds, such as allicin and sulfur-containing compounds, are believed to contribute to these effects. However, more large-scale human studies are needed to confirm its effectiveness.",
                    "sources": [
                        {
                            "name": "National Institutes of Health",
                            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7265806/"
                        },
                        {
                            "name": "Journal of Medicinal Food",
                            "url": "https://www.liebertpub.com/doi/10.1089/jmf.2018.0150"
                        }
                    ]
                }
            ]
        }
    
    return datasets

class MedVerifyAssistant:
    def __init__(self, datasets, rag_db):
        self.datasets = datasets
        self.rag_db = rag_db
        
        # Hugging Face API Configuration
        self.api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    def generate_response(self, prompt, retrieved_claims=None):
        """Generate a medical response using LLM with RAG support"""
        try:
            # Prepare context from RAG
            context = ""
            if retrieved_claims:
                context = "Retrieved medical evidence:\n\n"
                for idx, claim in enumerate(retrieved_claims):
                    context += f"Evidence #{idx+1} (Relevance: {claim.get('relevance_score', 0):.2f}):\n"
                    context += f"Claim: {claim.get('claim', '')}\n"
                    context += f"Evidence Level: {claim.get('evidence_level', '')}\n"
                    context += f"Explanation: {claim.get('explanation', '')}\n\n"

            # Construct the full prompt - we don't include the full prompt instructions in the output
            system_prompt = (
                "You are MedVerify, a professional medical knowledge assistant. "
                "Provide scientifically accurate and evidence-based responses. "
                "Use the retrieved evidence where relevant, but also apply medical knowledge "
                "to provide a comprehensive assessment.\n\n"
            )
            
            instruction_prompt = (
                f"Analyze the following medical claim: {prompt}\n\n"
                "Consider:\n"
                "1. Whether the claim appears to be supported by evidence\n"
                "2. The strength of available evidence\n"
                "3. Important context or caveats to consider\n"
                "4. Your overall assessment\n\n"
                f"{context}\n"
            )

            full_prompt = system_prompt + instruction_prompt
            
            # Call Hugging Face API
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                json={"inputs": full_prompt}
            )
            
            if response.status_code == 200:
                response_json = response.json()
                if isinstance(response_json, list) and response_json:
                    # Extract just the model's response, not including the prompt
                    generated_text = response_json[0].get("generated_text", "")
                    
                    # Find where the system and instruction prompts end and the actual response begins
                    # Removing the prompt part to just show the response
                    response_only = generated_text.replace(full_prompt, "").strip()
                    
                    # If we can't extract cleanly, at least make sure we don't repeat the query
                    if not response_only or prompt in response_only:
                        # As a fallback, get the last substantial part of the text
                        parts = generated_text.split('\n\n')
                        response_only = parts[-1] if parts else generated_text
                    
                    return response_only
                else:
                    return "The model did not return a proper response. Please try again."
            else:
                return f"API Error: {response.status_code}. Please ensure your HF_TOKEN is valid."

        except Exception as e:
            return f"Error generating response: {str(e)}"

class SimplifiedMedicalReportAnalyzer:
    def __init__(self):
        # Hugging Face API Configurations - using only 2 models as specified
        self.ner_model_url = "https://api-inference.huggingface.co/models/Helios9/BioMed_NER"
        self.llm_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    def test_model_access(self):
        """Check Hugging Face API accessibility with minimal test calls"""
        models = {
            "Medical Extractor": self.ner_model_url,
            "LLM": self.llm_url
        }
        
        for name, url in models.items():
            try:
                response = requests.post(url, headers=self.headers, json={"inputs": "Test"})
                
                if response.status_code == 200:
                    print(f"✅ {name}: Access successful")
                else:
                    print(f"❌ {name}: Error {response.status_code}")
            except Exception as e:
                print(f"❌ {name}: Exception - {str(e)}")

    
    def analyze_medical_report(self, pdf_file):
        """Simplified medical report analysis with minimal API calls"""
        try:
            self.test_model_access()
            # Extract text from PDF
            pdf_bytes = pdf_file.read()
            pdf_file_obj = tempfile.NamedTemporaryFile(delete=False)
            pdf_file_obj.write(pdf_bytes)
            pdf_file_obj.close()
            
            reader = PyPDF2.PdfReader(pdf_file_obj.name)
            full_text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            # Clean up temporary file
            os.unlink(pdf_file_obj.name)
            
            # Truncate text to avoid API limits
            truncated_text = full_text[:5000]  # Limiting to 5000 characters for API efficiency
            
            # STEP 1: Send raw text to the NER model
            response = requests.post(
                self.ner_model_url,
                headers=self.headers,
                json={"inputs": truncated_text}
            )


            if response.status_code != 200:
                st.error(f"NER model error: {response.status_code}")
                return None
                
            result = response.json()

            # Process the NER model results
            medical_terms = []
            try:
                # Handle flat list structure from BioMed_NER
                if isinstance(result, list):
                    for entity in result:
                        # Extract term directly from the entity object
                        term = entity.get("word", "").strip()
                        if term and len(term) > 3:  # Filter out very short terms
                            medical_terms.append(term)
                
                # Remove duplicates and limit to top 15 terms
                medical_terms = list(set(medical_terms))[:15]
            except Exception as e:
                st.error(f"Error processing NER response: {str(e)}")
                # Fallback: use LLM to extract terms if NER processing fails
                medical_terms = []
            
            # STEP 2: Single API call for explanations and summary
            explanation_prompt = (
                f"Task 1: For each of these medical terms, provide a 1-2 line simple explanation that a patient would understand:\n{', '.join(medical_terms)}\n\n"
                f"Task 2: Provide a 2-3 paragraph summary of the following medical report in patient-friendly language:\n{truncated_text}\n\n"
                "Format your response as:\n"
                "MEDICAL TERMS EXPLAINED:\n"
                "[Term]: [Simple explanation]\n"
                "...\n\n"
                "REPORT SUMMARY:\n"
                "[Your 2-3 paragraph summary]"
                "Don't rephrase the prompt"
            )
            
            # Make the second API call
            response = requests.post(
                self.llm_url,
                headers=self.headers,
                json={"inputs": explanation_prompt}
            )

            st.write("API Response:", response.json())  # Debugging

            
            if response.status_code != 200:
                st.error(f"LLM model error: {response.status_code}")
                return None
                
            result = response.json()
            if not isinstance(result, list) or not result:
                st.error("Invalid response from LLM model")
                return None
                
            raw_text = result[0].get("generated_text", "")
            match = re.search(r"MEDICAL TERMS EXPLAINED:\s*(.*)", raw_text, re.DOTALL)
            return match.group(1).strip()
            
        except Exception as e:
            st.error(f"Error analyzing medical report: {str(e)}")
            return None

# UI Layout and Application Logic
def main():
    # Configure Streamlit page
    st.title("MedGuide")
    st.markdown("### 🩺 Medical Information Verifier & Report Explainer")
    
    # Load datasets
    datasets = load_datasets()
    
    # Create RAG database
    rag_db = RAGDatabase()
    num_claims = rag_db.load_healthfc_data(datasets['healthfc'])
    
    # Initialize components
    assistant = MedVerifyAssistant(datasets, rag_db)
    report_analyzer = SimplifiedMedicalReportAnalyzer()
    
    # API token status check in sidebar
    token_valid = Config.validate_token()
    
    # Sidebar information
    st.sidebar.title("About MedGuide")
    st.sidebar.info(
        "MedGuide uses AI to help verify medical claims and explain medical reports in simple language. "
        f"Currently loaded with {num_claims} health claims from HealthFC database."
    )
    
    # Set up tabs for different functionalities
    tab1, tab2 = st.tabs(["Verify Medical Claims", "Explain Medical Reports"])
    
    # Tab 1: Medical Claim Verification
    with tab1:
        st.subheader("Medical Claim Verification")
        st.markdown(
            "Submit a medical claim or statement to check its validity against our database."
        )
        
        # Medical claim input
        claim = st.text_area("Enter a medical claim or question:", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            search_k = st.slider("Number of reference sources:", 1, 5, 3)
        
        # Process button
        if st.button("Verify Claim", key="verify_claim"):
            if not claim:
                st.warning("Please enter a medical claim to verify.")
            elif not token_valid:
                st.error("Please set your Hugging Face API token to continue.")
            else:
                with st.spinner("Analyzing claim..."):
                    # Search for relevant claims
                    retrieved_claims = rag_db.search_claims(claim, top_k=search_k)
                    
                    # Generate response
                    response = assistant.generate_response(claim, retrieved_claims)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.markdown(response)
                    
                    # Display references
                    if retrieved_claims:
                        st.subheader("Reference Sources")
                        for i, ref in enumerate(retrieved_claims):
                            with st.expander(f"Reference {i+1}: {ref.get('claim', 'Unknown Claim')}"):
                                st.markdown(f"**Evidence Level**: {ref.get('evidence_level', 'Not specified')}")
                                st.markdown(f"**Explanation**: {ref.get('explanation', 'No explanation available.')}")
                                st.markdown("**Sources**:")
                                for source in ref.get('sources', []):
                                    st.markdown(f"- [{source.get('name', 'Source')}]({source.get('url', '#')})")
                                st.markdown(f"**Relevance Score**: {ref.get('relevance_score', 0):.2f}")
    
    # Tab 2: Medical Report Explanation
    with tab2:
        st.subheader("Medical Report Explanation")
        uploaded_file = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])

        if uploaded_file is not None:
            try:
                if not token_valid:
                    st.error("Please set your Hugging Face API token to continue.")
                else:
                    with st.spinner("Analyzing medical report... This may take a minute."):
                        report_summary = report_analyzer.analyze_medical_report(uploaded_file)

                        if report_summary:
                            st.subheader("Report Summary in Simple Terms")
                            st.markdown(report_summary)
                        else:
                            st.error("Could not analyze the medical report. Please check if the file is a valid PDF.")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()