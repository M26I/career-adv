import streamlit as st
import pandas as pd
import re
import nltk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.skill_matcher import match_resume_to_job, get_job_embeddings



# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load job titles and skills from your CSVs
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
skills_df = pd.read_csv("data/cleaned_skills.csv")
known_skills = set(skills_df['skill'].dropna().str.lower())

job_titles = set(job_titles_df['job_title'].str.lower().dropna().unique())

for skills in skills_df['skill'].dropna():
    known_skills.update([skill.strip().lower() for skill in skills.split(',')])

# Function to extract skills from resume (optional, if needed later)
def extract_skills(resume_text):
    resume_text = resume_text.lower()
    found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text)]
    return list(set(found_skills))

# Streamlit app
def main():
    st.title("AI Career Advisor - PathFinder")

    st.write(
        "Upload a resume (in PDF or text format), and this app will suggest the most relevant job title based on your skills and experience."
    )

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Process the uploaded file
        if uploaded_file.type == "application/pdf":
            # If the uploaded file is a PDF, read it
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = ""
            for page in reader.pages:
                resume_text += page.extract_text()
        elif uploaded_file.type == "text/plain":
            # If the uploaded file is plain text
            resume_text = str(uploaded_file.read(), "utf-8")
        
        # Extract skills from the resume (if needed)
        extracted_skills = extract_skills(resume_text)

        # Use the resume text to match to the best job title
        job_embeddings = get_job_embeddings(list(job_titles))

        best_job_title, similarity_score = match_resume_to_job(resume_text, job_embeddings)

        # Display the results
        st.subheader("Extracted Information")
        
        st.write("### Skills Extracted")
        st.write(extracted_skills)

        st.write("### Recommended Job Title")
        st.write(f"**{best_job_title}**")
        st.write(f"Similarity Score: {similarity_score:.4f}")

if __name__ == "__main__":
    main()
