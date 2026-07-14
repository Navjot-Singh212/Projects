# The Code Kitchen

Welcome to my central project hub! I am a B.Tech Computer Science & Engineering student specializing in Machine Learning and Data Science.

This repository serves as the projects archive for my technical journey. Inside, you will find all the projects I have made.

Feel free to explore the folders below to see what I have been building!

---

##  Featured Projects

### 1. AI-Based Resume Analysis System
**Status:** Live & Deployed  

####  What It Does
This is an AI recruiting tool designed to calculate mathematically accurate ATS (Applicant Tracking System) scores. Instead of relying on flawed keyword-matching, this application uses dense vector embeddings and cosine similarity to understand the deep semantic context of both a candidate's resume and a job description. 

Key features include:
*   **Contextual Fluff Filtering:** Automatically identifies and ignores HR buzzwords and corporate fluff in job descriptions to focus purely on technical requirements.
*   **AI Skill Gap Analysis:** Integrates with the Gemini LLM to generate personalized, actionable feedback on missing hard skills.
*   **Vision-Based OCR Fallback:** Automatically detects corrupted or unreadable PDFs and triggers an AI vision scanner to extract text directly from the document images.
*   **Live Web Dashboard:** A clean, responsive UI deployed via Streamlit Community Cloud.

#### Tech Stack & Tools
*   **Language:** Python
*   **Frontend & Hosting:** Streamlit, Streamlit Community Cloud
*   **Machine Learning (NLP):** `SentenceTransformers` (all-MiniLM-L6-v2), `scikit-learn` (Cosine Similarity)
*   **Generative AI:** Google Gemini API (`google-generativeai`)
*   **Data Processing:** `PyMuPDF` (fitz), `Pillow` (Image Processing), `NLTK`, `ftfy`


---

*(More projects coming soon...)*
