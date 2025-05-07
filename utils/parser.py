import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Only needed once

# Load job titles and skills from your CSVs
job_titles_df = pd.read_csv("data/all_job_post.csv")
skills_df = pd.read_csv("data/all_job_post.csv")

job_titles = set(job_titles_df['job_title'].str.lower().dropna().unique())
known_skills = set()
for skills in skills_df['job_skill_set'].dropna():
    known_skills.update([skill.strip().lower() for skill in skills.split(',')])

def extract_skills_and_titles(resume_text):
    resume_text = resume_text.lower()
    words = word_tokenize(resume_text)

    # Match job titles
    found_titles = [title for title in job_titles if title in resume_text]

    # Match skills
    found_skills = [skill for skill in known_skills if skill in resume_text]

    return {
        "job_titles": list(set(found_titles)),
        "skills": list(set(found_skills))
    }

# Example usage
if __name__ == "__main__":
    sample_text = """
    Experienced in Python, SQL, and Tableau. Worked as a Data Analyst at a fintech company.
    Familiar with Machine Learning and Pandas. Seeking roles in AI or Data Science.
    """
    result = extract_skills_and_titles(sample_text)
    print(result)
