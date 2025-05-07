import pandas as pd

# Load raw job titles
df = pd.read_csv("data/job_postings.csv")

# Lowercase and drop missing
titles = df['job_title'].dropna().str.strip().str.lower()

# Filter out too short titles or generic words
banned_words = ['a', 'an', 'the', 'at', 'in', 'on', 'of', 'job', 'staff', 'team', 'work', 'employee', 'opening']

def is_valid_title(title):
    if len(title) <= 3:
        return False
    if any(word in title.split() for word in banned_words):
        return True  # Still allow longer titles with mixed words
    return True

# Filter and deduplicate
cleaned_titles = titles[titles.apply(is_valid_title)].drop_duplicates()

# Save to a new file
cleaned_titles_df = pd.DataFrame(cleaned_titles, columns=["job_title"])
cleaned_titles_df.to_csv("data/cleaned_job_titles.csv", index=False)

print(f"Saved {len(cleaned_titles_df)} cleaned job titles.")
