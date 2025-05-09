# AI Career Advisor – PathFinder

**PathFinder** is an AI-powered resume analyzer that helps you uncover career paths aligned with your skills. PathFinder provides personalized job title suggestions, skill insights, and improvement recommendations — all in one streamlined app.

## 🔍 What It Does

Upload your own resume or choose from one of six provided samples to receive a detailed analysis:

- ✅ **Top 5 matching job titles**, including:
  - Role category (e.g., Technical, Design-focused)
  - Matching score
  - Key required skills
- 🔎 **Job titles found** in your resume (if any)
- 🧠 **Skills extracted** from your resume
- 📈 **Suggested skills** to learn or add for better alignment

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) – UI and interactivity
- [Python](https://www.python.org/)
- [NLTK](https://www.nltk.org/) – Natural language processing
- [pandas](https://pandas.pydata.org/) – Data manipulation
- [sentence-transformers](https://www.sbert.net/) – Text embeddings for similarity scoring


## 📦 Dataset

The app uses the [Data Science Job Postings & Skills (2024)](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills?select=job_skills.csv) dataset from Kaggle to extract known job titles and skills for comparison.

## 📁 Sample Resumes Included

Explore sample career paths using built-in resumes for:

- Cloud Engineer  
- Frontend Developer  
- Cyber Security Engineer  
- Data Scientist  
- Product Manager  
- UX Designer  

These are located in the `sample_resumes/` folder and can be selected directly in the app.

## 🚀 Getting Started

1. **Clone the repo**:
   ```bash
   git clone https://github.com/M26I/career-adv
   cd career-adv

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the app**:
   ```bash
   streamlit run app/app.py

4. **Open the Streamlit interface**: 
   Localhost  http://localhost:8501


## ⚠️ Limitations ##

-The app suggests job titles only based on the dataset — it does not scrape external job sources.

-Resume analysis may take a few seconds due to embedding computations.

## Author ##
[M26I](https://github.com/M26I)
