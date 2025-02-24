from fastapi import FastAPI, UploadFile, File, Form
import pdfplumber
import docx
import requests  # Use Groq API
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hugging_face="use your hugging face api key"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", use_auth_token=hugging_face)


GROQ_API_KEY = "use your groq api key"


client = MongoClient("use your mongodb url")
db = client["resume_screening"]
resumes_collection = db["resumes"]


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return " ".join([para.text for para in doc.paragraphs])

# Function to call DeepSeek LLM via Groq API
def extract_skills_from_groq(resume_text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": f"Extract key skills from this resume: {resume_text}"}]
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_data = response.json()
    
    # Extract skills from the API response
    if "choices" in response_data and response_data["choices"]:
        return response_data["choices"][0]["message"]["content"]
    return "Error extracting skills"

@app.post("/screen_resume/")
async def screen_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    # Determine file type and extract text
    if file.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(file.file)
    elif file.filename.endswith(".docx"):
        resume_text = extract_text_from_docx(file.file)
    else:
        return {"error": "Unsupported file format"}

  
    skills = extract_skills_from_groq(resume_text)

  
    resume_embedding = model.encode(skills, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    # Compute similarity score
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

    # Store in MongoDB
    resume_entry = {
        "name": file.filename,
        "skills": skills,
        "similarity_score": similarity_score
    }
    resumes_collection.insert_one(resume_entry)

    return {"resume_text": resume_text, "extracted_skills": skills, "similarity_score": similarity_score}
