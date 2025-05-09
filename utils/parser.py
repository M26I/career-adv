import re

def extract_skills_and_titles(resume_text, job_titles, known_skills):
    resume_text = resume_text.lower()

    found_titles = [title for title in job_titles if re.search(r'\b' + re.escape(title) + r'\b', resume_text)]
    found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text)]

    return {
        "job_titles": list(set(found_titles)),
        "skills": list(set(found_skills))
    }
