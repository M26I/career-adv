from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# Load job titles
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
job_titles = job_titles_df['job_title'].dropna().unique().tolist()  # Convert to list

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_job_embeddings(job_titles_list):
    """Generate embeddings for each job title"""
    return model.encode(job_titles_list, show_progress_bar=True, convert_to_tensor=True)

def match_resume_to_job(resume_text, job_titles_list, job_embeddings, top_k=5):
    """Match resume to job titles using cosine similarity"""
    # Get resume embedding
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.cos_sim(resume_embedding, job_embeddings)[0]

    # Get top-k job indices
    top_results = torch.topk(similarities, k=top_k)

    # Format results
    matches = []
    for idx, score in zip(top_results.indices, top_results.values):
        matches.append({
            "job_title": job_titles_list[idx],
            "score": round(float(score), 3)
        })

    return matches

# Optional CLI test
if __name__ == "__main__":
    sample_resume = """
    Experienced in Python, SQL, and Tableau. Worked as a Data Analyst at a fintech company.
    Familiar with Machine Learning and Pandas. Seeking roles in AI or Data Science.
    """

    embeddings = get_job_embeddings(job_titles)
    top_matches = match_resume_to_job(sample_resume, job_titles, embeddings)

    for match in top_matches:
        print(f"{match['job_title']} (score: {match['score']})")
