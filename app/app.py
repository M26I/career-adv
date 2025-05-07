import os
from flask import Flask, request, jsonify
from docx import Document
import PyPDF2
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx'}

# Load the job titles and skills
job_titles_df = pd.read_csv("data/cleaned_job_titles.csv")
skills_df = pd.read_csv("data/cleaned_skills.csv")
known_skills = set(skills_df['skill'].dropna().str.lower())

job_titles = set(job_titles_df['job_title'].str.lower().dropna().unique())

for skills in skills_df['skill'].dropna():
    known_skills.update([skill.strip().lower() for skill in skills.split(',')])

# Function to extract skills and titles from resume text
def extract_skills_and_titles(resume_text):
    resume_text = resume_text.lower()

    found_titles = [title for title in job_titles if re.search(r'\b' + re.escape(title) + r'\b', resume_text)]
    found_skills = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill) + r'\b', resume_text)]

    return {
        "job_titles": list(set(found_titles)),
        "skills": list(set(found_skills))
    }

# Helper function to extract text from docx
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Helper function to extract text from pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Helper function to extract text from txt file
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to upload resume
@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Extract text based on file type
        if file.filename.endswith('.txt'):
            resume_text = extract_text_from_txt(filename)
        elif file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(filename)
        elif file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(filename)

        # Pass the resume text to your parser
        result = extract_skills_and_titles(resume_text)

        return jsonify({'message': 'File successfully uploaded', 'filename': file.filename, 'parsed_result': result}), 200
    else:
        return jsonify({'error': 'Invalid file format. Only .txt, .pdf, .docx are allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
