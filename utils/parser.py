import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')  # Only needed once

# Load job titles and skills from your CSVs
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
skills_df = pd.read_csv("data/cleaned_skills.csv")

# Prepare sets
job_titles = set(job_titles_df['job_title'].str.lower().dropna().unique())
known_skills = set(skills_df['skill'].dropna().str.lower())

def extract_skills_and_titles(resume_text):
    resume_text = resume_text.lower()

    # Use word boundary matching to avoid partial matches
    found_titles = [title for title in job_titles if re.search(r'\b' + re.escape(title) + r'\b', resume_text)]
    found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text)]

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
