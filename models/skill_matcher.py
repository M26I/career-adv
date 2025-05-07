from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the job titles and skills data
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
job_titles = job_titles_df['job_title'].dropna().unique()

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_job_embeddings(job_titles):
    """Generate embeddings for each job title"""
    return model.encode(job_titles, show_progress_bar=True)

def match_resume_to_job(resume_text, job_embeddings):
    """Match resume text to job titles based on cosine similarity"""
    # Get the embedding for the resume text
    resume_embedding = model.encode([resume_text])

    # Compute cosine similarity between resume and job titles
    similarities = cosine_similarity(resume_embedding, job_embeddings)
    
    # Get the index of the best matching job
    best_match_idx = similarities.argmax()
    best_job_title = job_titles[best_match_idx]

    return best_job_title, similarities[0][best_match_idx]

if __name__ == "__main__":
    # Example usage with a sample resume text
    sample_resume = """
    Experienced in Python, SQL, and Tableau. Worked as a Data Analyst at a fintech company.
    Familiar with Machine Learning and Pandas. Seeking roles in AI or Data Science.
    """
    
    job_embeddings = get_job_embeddings(job_titles)
    best_match_job, similarity_score = match_resume_to_job(sample_resume, job_embeddings)
    
    print(f"Best Match Job: {best_match_job}")
    print(f"Similarity Score: {similarity_score}")
