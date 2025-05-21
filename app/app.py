# © 2025 M26I - For educational/portfolio use only

import streamlit as st
import pandas as pd
import re
import nltk
import sys
import os
from collections import Counter
from difflib import SequenceMatcher

# Ensure the 'models' and 'utils' folders are in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.skill_matcher import match_resume_to_job, get_job_embeddings
from utils.parser import extract_skills_and_titles

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load job titles and skills once
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
skills_df = pd.read_csv("data/cleaned_skills.csv")

job_titles = set(job_titles_df['job_title'].dropna().str.lower().unique())
known_skills = set(skills_df['skill'].dropna().str.lower())

# Define broad categories for highlighting
broad_categories = {
    "Technical": ["developer", "engineer", "scientist", "devops", "reliability", "cloud"],
    "Design-focused": ["designer", "ux", "ui"],
    "Business-oriented": ["manager", "analyst", "owner", "strategist", "product"]
}

category_colors = {
    "Technical": "#d1e7dd",
    "Design-focused": "#cfe2ff",
    "Business-oriented": "#fff3cd",
    "General": "#e7eaf6"
}

def categorize_job_title(title):
    title = title.lower()
    for category, keywords in broad_categories.items():
        if any(k in title for k in keywords):
            return category
    return "General"

def suggest_missing_skills(resume_text, extracted_skills, known_skills, top_n=10):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    words = re.findall(r'\b\w+\b', resume_text.lower())
    word_counts = Counter(words)

    skill_scores = []
    for skill in known_skills:
        if skill in extracted_skills:
            continue
        if len(skill.split()) > 2:
            continue
        if all(word in stop_words for word in skill.split()):
            continue

        relevance = sum(word_counts.get(word, 0) for word in skill.split())
        if relevance > 0:
            skill_scores.append((skill, relevance))

    sorted_skills = sorted(skill_scores, key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills[:top_n]]

# Main app
def main():
    st.title("AI Career Advisor - PathFinder")
    st.write("Choose a sample or upload a resume to get the top 5 matching job titles based on your skills.")

    sample_resume_folder = "sample_resumes"
    os.makedirs(sample_resume_folder, exist_ok=True)
    sample_files = [f for f in os.listdir(sample_resume_folder) if f.endswith((".txt", ".pdf"))]
    sample_choice = st.selectbox("Choose a sample resume:", ["-- Select --"] + sample_files)

    uploaded_file = None
    resume_text = ""

    if sample_choice != "-- Select --":
        sample_path = os.path.join(sample_resume_folder, sample_choice)
        if sample_path.endswith(".txt"):
            with open(sample_path, "r", encoding="utf-8") as f:
                resume_text = f.read()
        elif sample_path.endswith(".pdf"):
            import PyPDF2
            with open(sample_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                resume_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    else:
        uploaded_file = st.file_uploader("Or upload your own resume", type=["pdf", "txt"])
        if uploaded_file is not None:
            if uploaded_file.type == "application/pdf":
                import PyPDF2
                reader = PyPDF2.PdfReader(uploaded_file)
                resume_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            elif uploaded_file.type == "text/plain":
                resume_text = str(uploaded_file.read(), "utf-8")

    if resume_text:
        extracted = extract_skills_and_titles(resume_text, job_titles, known_skills)
        extracted_skills = extracted["skills"]
        extracted_titles = extracted["job_titles"]
        suggested_skills = suggest_missing_skills(resume_text, extracted_skills, known_skills)

        job_embeddings = get_job_embeddings(list(job_titles))
        top_matches = match_resume_to_job(resume_text, list(job_titles), job_embeddings, top_k=5)

        st.subheader("Results")
        st.write("### Top 5 Job Matches")
        for match in top_matches:
            category = categorize_job_title(match['job_title'])
            bg_color = category_colors.get(category, "#e2e3e5")
            st.markdown(
                f"<div style='background-color:{bg_color}; padding:10px; border-radius:6px; margin-bottom:6px;'>"
                f"<strong>{match['job_title'].title()}</strong> — <em>Category:</em> {category} — <strong>Score:</strong> {match['score']:.3f}"
                f"</div>",
                unsafe_allow_html=True
            )

        st.write("### Job Titles Found in Resume")
        st.write(extracted_titles if extracted_titles else "No job titles detected.")

        st.write("### Skills Extracted from Your Resume")
        if extracted_skills:
            cols = st.columns(min(len(extracted_skills), 5))
            for i, skill in enumerate(extracted_skills):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"<div style='background-color:#f0f2f6; padding:8px; border-radius:5px; text-align:center;'>{skill.title()}</div>",
                        unsafe_allow_html=True
                    )
        else:
            st.write("No clear skills found.")

        st.write("### Suggested Skills to Learn or Add")
        st.write(suggested_skills if suggested_skills else "No suggestions — great coverage!")

if __name__ == "__main__":
    main()
