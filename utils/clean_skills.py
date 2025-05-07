import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load the stopwords
stop_words = set(stopwords.words('english'))

# Load the job_skills.csv file
df = pd.read_csv("data/job_skills.csv")

# Create an empty set for clean skills
clean_skills = set()

# Loop through each skill entry
for skills in df['job_skills'].dropna():
    for skill in skills.split(','):
        skill = skill.strip().lower()
        if len(skill) > 2 and skill not in stop_words:
            clean_skills.add(skill)

# Save cleaned skills to a new CSV
cleaned_df = pd.DataFrame(sorted(clean_skills), columns=["skill"])
cleaned_df.to_csv("data/cleaned_skills.csv", index=False)

print(f"✅ Cleaned {len(clean_skills)} unique skills.")
