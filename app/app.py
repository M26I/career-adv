import streamlit as st
import pandas as pd
import re
import nltk
import sys
import os

# Ensure the 'models' folder is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.skill_matcher import match_resume_to_job, get_job_embeddings

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load job titles and skills
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
skills_df = pd.read_csv("data/cleaned_skills.csv")
known_skills = set(skills_df['skill'].dropna().str.lower())

job_titles = job_titles_df['job_title'].dropna().str.lower().unique().tolist()

for skills in skills_df['skill'].dropna():
    known_skills.update([skill.strip().lower() for skill in skills.split(',')])

# Function to extract skills from resume
def extract_skills(resume_text):
    resume_text = resume_text.lower()
    found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text)]
    return list(set(found_skills))

# Streamlit app
def main():
    st.title("AI Career Advisor - PathFinder")

    st.write(
        "Upload a resume (PDF or TXT), and get the top 5 job titles that best match your skills and experience."
    )

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif uploaded_file.type == "text/plain":
            resume_text = str(uploaded_file.read(), "utf-8")

        # Extract skills
        extracted_skills = extract_skills(resume_text)

        # Generate embeddings
        job_embeddings = get_job_embeddings(job_titles)

        # Match resume to top job titles
        top_matches = match_resume_to_job(resume_text, job_titles, job_embeddings, top_k=5)

        # Display results
        st.subheader("Extracted Information")
        st.write("### Skills Extracted")
        st.write(extracted_skills if extracted_skills else "No clear skills found.")

        st.write("### Top 5 Job Matches")
        for match in top_matches:
            st.write(f"**{match['job_title']}** — Similarity Score: {match['score']:.3f}")

if __name__ == "__main__":
    main()
