#AI based resume analysis system

import fitz
import re
import unicodedata
import ftfy
import io
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
from google.api_core import exceptions
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


st.set_page_config(page_title="AI Resume Analysis System",page_icon="📄",layout="wide")
st.title("AI Resume Analysis System")
st.markdown("Upload your Resume PDF and paste your Job description to get a mathematically accurate ATS score and AI Skill gap analysis~")
@st.cache_resource
def load_models():
    load_dotenv()
    api_key=os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model=SentenceTransformer("all-MiniLM-L6-v2")
    return model

semantic_Model=load_models()

#the pre-processing and cleaning segment~

def full_text_processing(rawstr):

    rawstr=rawstr.lower()
    rawstr=ftfy.fix_text(rawstr)
    rawstr=unicodedata.normalize("NFKC",rawstr)
    cleanedstr=re.sub(r"[^a-z0-9\s.,\-]"," ",rawstr)
    cleanedstr=re.sub(r" +"," ",cleanedstr)
    cleanedstr=re.sub(r"\x00","",cleanedstr)
    tokenisedsentences=re.split(r"[\n.]+",cleanedstr)
    cleanTokenisedSentences=[line.strip() for line in tokenisedsentences if line.strip()!=""]

    return cleanTokenisedSentences



#the ORS fallback segment~

def is_text_corrupted(textfile):
    #this function evaluates whether the pdf extraction fails or not
    if (len(textfile.strip())==0):
        return True
    
    if (re.search(r'\(cid:\d+\)',textfile)):
        return True
    
    if textfile.count("\ufffd")>3 or textfile.count("Ɵ")>2:
        return True
    
    alphanumericCount=sum(1 for char in textfile if char.isalnum())
    ReadableRatio=alphanumericCount/len(textfile)
    if ReadableRatio<0.50:
        return True
    
    return False




def extract_Via_OCR(textfile_bytes):
    #this function will trigger the OCR fallback~

    extracted_text=""
    model=genai.GenerativeModel("gemini-3.1-flash-lite")
    
    with fitz.open(stream=textfile_bytes,filetype="pdf") as pdf:
        for page in range(len(pdf)):
            currentpage=pdf[page]

            zoomed_matrix=fitz.Matrix(2,2)
            pix=currentpage.get_pixmap(matrix=zoomed_matrix)
            
            img_data=pix.tobytes("png")
            Img=Image.open(io.BytesIO(img_data))

            prompt = ("You are an OCR scanner. Extract absolutely all the text from this resume image exactly as it is written. Do not summarize, do not format with markdown, and do not add any conversational text.")
            st.toast(f"scanning page {page+1} with AI vision~")
            try:
                response=model.generate_content([prompt,Img])
                extracted_text+=response.text+"\n"
                time.sleep(15)

            except exceptions.ResourceExhausted as e:
                st.error(f"ERROR: Quota Exceeded on page {page+1}, try again tomorrow~")
                break
            
            except Exception as e:
                st.error("ERROR: an error has appeared~!!")
                st.error(f"DETAILS:{e}")
                break

            
    st.success("AI vision successfully extracted the text!")
    return extracted_text




#Job Description cleaning~
def cleanJD(jd,targetRole):
    skillAnchor=f"Core skills, soft skills, experience, knowledge and requirements for a {targetRole}."
    responsibiltitesAnchor="Day-to-day duties, project execution, tasks, job responsibilities, role expectations"
    fluffAnchor="Company perks, health insurance, salary, office environment, dental plan, equity, PTO, startup culture,fun,random"
    eduAnchor="University degrees, B.S., Master's, PhD, professional certifications, academic qualifications."
    softskillAnchor="Communication, leadership, teamwork, problem-solving, time management, interpersonal skills."

    MatrixSkillAnchor=semantic_Model.encode([skillAnchor])
    MatrixrespoAnchor=semantic_Model.encode([responsibiltitesAnchor])
    MatrixEDUAnchor=semantic_Model.encode([eduAnchor])
    MatrixSSAnchor=semantic_Model.encode([softskillAnchor])
    MatrixFluffAnchor=semantic_Model.encode([fluffAnchor])

    cleanJobSentences=[]
    for sentence in jd:
        matrix_JD_sentence=semantic_Model.encode([sentence])

        skillscore=cosine_similarity(matrix_JD_sentence,MatrixSkillAnchor)[0][0]
        Resposcore=cosine_similarity(matrix_JD_sentence,MatrixrespoAnchor)[0][0]
        EDUscore=cosine_similarity(matrix_JD_sentence,MatrixEDUAnchor)[0][0]
        SSscore=cosine_similarity(matrix_JD_sentence,MatrixSSAnchor)[0][0]
        fluffscore=cosine_similarity(matrix_JD_sentence,MatrixFluffAnchor)[0][0]

        highestScore=max(skillscore,SSscore,Resposcore,EDUscore)
        margin_of_error=0.05
        if fluffscore> (highestScore+margin_of_error):
            pass
        else:
            cleanJobSentences.append(sentence)
        

    return cleanJobSentences

# the main PDF-text exctraction~

def pdfTextExtraction_and_Processing(inputPDF):
    fullRawString=""
    pdf_bytes=inputPDF.read()
    with fitz.open(stream=pdf_bytes,filetype="pdf") as pdf:
        length=len(pdf)
        for pageNum in range(length):
            currentPage=pdf[pageNum]
            pageText=currentPage.get_text()+"\n"
            if pageText:
                fullRawString+=pageText
        

    if(is_text_corrupted(fullRawString)):
        st.error(f"Text Corruption detected, triggering OCR fallback!!!")
        fullRawString=""
        fullRawString=extract_Via_OCR(pdf_bytes)

        
    else:
        st.success("Text Extraction was successfull~")

    cleanFullTokens=full_text_processing(fullRawString)
    return cleanFullTokens

col1,col2=st.columns(2)
with col1:
    st.subheader("The Candidate")
    uploaded_file=st.file_uploader("Upload Resume (PDF)~",type=["pdf"])

with col2:
    st.subheader("The Job Role")
    jd_input=st.text_area("Print your Job Description here~",height=200)
    target_role=st.text_input("Enter the target Role (for example: Java Developer, Machine Learning Engineer etc.): ")

if st.button("Run ATS Analyis",use_container_width=True):
    if uploaded_file and jd_input and target_role:
        with st.spinner("Analysing Resume and Job Description..."):

            candidate1ResumeData=pdfTextExtraction_and_Processing(uploaded_file)
            jd=full_text_processing(jd_input)
            CleanedJD=cleanJD(jd,target_role)

            #Semantic Engine~
            resume_embeddings=semantic_Model.encode(candidate1ResumeData)
            JD_embeddings=semantic_Model.encode(CleanedJD)
            finalScoreMatrix=cosine_similarity(JD_embeddings,resume_embeddings)  # A Matrix filled with scores Rows= JD sentences and columns are Resume sentences

            #calculating the final score
            failed_JD_requirements=[]
            ATS_threshold=0.45
            passed_requirements=0
            totalRequirements=len(CleanedJD)
            for i,row in enumerate(finalScoreMatrix):
                highest_score=row.max()
                if highest_score>ATS_threshold:
                    passed_requirements+=1
    
                else:
                    failed_JD_requirements.append(CleanedJD[i])

            
            ats_score=(passed_requirements/totalRequirements)*100
            st.divider()
            st.header("Match Results~")
            st.metric(label="ATS Match Percentage: ",value=f"{ats_score:.2f}%")

            


            #skill Gap engine
            if len(failed_JD_requirements)>0:
                print("Analyzing skill gaps... ")
                missingConcepts="\n".join(failed_JD_requirements)

                analysis_prompt=analysis_prompt = f"""
                You are an expert technical recruiter analyzing why a candidate failed a resume screening.
                The candidate failed to match these specific sentences from the Job Description:
    
                {missingConcepts}
    
                First, completely ignore any sentences in that list that are just corporate culture fluff, perks, or HR buzzwords. 
                Then, based ONLY on the actual technical requirements provided, provide:
                1. Missing Keywords: A comma-separated list of the hard skills, frameworks, or tools the candidate lacks. If there are no technical skills missing, output "None".
                2. Improvement Plan: Two actionable sentences advising the candidate on how to bridge this gap. If no technical skills are missing, say "Your technical profile is a strong match."
                
                CRITICAL INSTRUCTION: Do not include any introductory text, conversational filler, or explanations of what you filtered out. Output ONLY the two numbered points exactly as requested.
                
              """
                

                st.subheader("AI Skill Gap Analysis~")
                model=genai.GenerativeModel("gemini-3.1-flash-lite")
                response=model.generate_content(analysis_prompt)
                st.info(response.text)

            else:
                st.success("AI-EVALUATION:")
                st.success("Candidate meets all the core Requirements, No critical skill gap detected~")
    else:
        st.error("Please Upload Resume PDF and paste the Job Description!!")

        





