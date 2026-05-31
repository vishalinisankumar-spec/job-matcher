import streamlit as st
import PyPDF2
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Smart Job Matcher", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { padding-top: 0; }
    .job-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .match-score {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Smart Job Matcher")
st.markdown("AI-Powered Resume Analysis & Job Recommendations")

with st.sidebar:
    st.header("📋 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=['pdf', 'txt'])
    
    st.markdown("---")
    st.header("⚙️ Settings")
    job_titles = st.multiselect(
        "Job titles to search:",
        ["Data Scientist", "Python Developer", "Backend Engineer", 
         "Machine Learning Engineer", "Data Engineer", "Full Stack Developer"],
        default=["Python Developer", "Data Scientist"]
    )
    
    locations = st.multiselect(
        "Preferred locations:",
        ["Remote", "New York", "San Francisco", "London", "Bangalore"],
        default=["Remote"]
    )

def extract_resume_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return file.read().decode('utf-8')

def parse_resume(text):
    resume_data = {
        'skills': [],
        'experience': [],
        'education': [],
        'full_text': text
    }
    
    skills_list = ['Python', 'Java', 'JavaScript', 'SQL', 'AWS', 'Docker', 'Kubernetes',
                   'React', 'Django', 'FastAPI', 'Flask', 'PostgreSQL', 'MongoDB',
                   'Machine Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'Scikit-learn',
                   'Data Analysis', 'API', 'REST', 'Git', 'Linux', 'Excel']
    
    for skill in skills_list:
        if skill.lower() in text.lower():
            resume_data['skills'].append(skill)
    
    years_match = re.findall(r'(\d+)\+?\s+years?\s+(?:of\s+)?experience', text, re.IGNORECASE)
    if years_match:
        resume_data['years_experience'] = max([int(y) for y in years_match])
    else:
        resume_data['years_experience'] = 0
    
    degrees = ['Bachelor', 'Master', 'PhD', 'B.S', 'M.S', 'B.A', 'M.A', 'MBA']
    for degree in degrees:
        if degree in text:
            resume_data['education'].append(degree)
    
    return resume_data

def get_sample_jobs(titles, locations):
    jobs = [
        {
            'title': 'Senior Python Developer',
            'company': 'Tech Corp',
            'location': 'Remote',
            'salary': '$120K-$160K',
            'description': 'Looking for experienced Python developer. Django, FastAPI, PostgreSQL required.',
            'skills_required': ['Python', 'Django', 'PostgreSQL', 'Docker', 'Git'],
        },
        {
            'title': 'Data Scientist',
            'company': 'AI Innovations',
            'location': 'San Francisco',
            'salary': '$130K-$180K',
            'description': 'Build ML models. Python, scikit-learn, TensorFlow required.',
            'skills_required': ['Python', 'Machine Learning', 'TensorFlow', 'Pandas', 'SQL'],
        },
        {
            'title': 'Backend Engineer',
            'company': 'StartUp Inc',
            'location': 'Remote',
            'salary': '$100K-$140K',
            'description': 'Build scalable APIs. FastAPI and AWS experience required.',
            'skills_required': ['Python', 'FastAPI', 'AWS', 'PostgreSQL', 'Docker'],
        },
        {
            'title': 'Machine Learning Engineer',
            'company': 'ML Systems',
            'location': 'New York',
            'salary': '$140K-$200K',
            'description': 'Develop production ML systems. Python and ML knowledge required.',
            'skills_required': ['Python', 'PyTorch', 'TensorFlow', 'SQL', 'Docker'],
        },
    ]
    
    filtered_jobs = [j for j in jobs if j['location'] in locations or 'Remote' in locations]
    return filtered_jobs

def calculate_match_score(resume_data, job):
    match_score = 0
    
    skills_matched = set(resume_data['skills']) & set(job['skills_required'])
    skills_match_pct = (len(skills_matched) / len(job['skills_required'])) * 100 if job['skills_required'] else 0
    match_score += skills_match_pct * 0.4
    
    if resume_data['years_experience'] >= 3:
        match_score += 30
    elif resume_data['years_experience'] >= 1:
        match_score += 15
    else:
        match_score += 5
    
    if resume_data['education']:
        match_score += 20
    
    return min(match_score, 100)

if uploaded_file:
    with st.spinner("🔄 Processing your resume..."):
        resume_text = extract_resume_text(uploaded_file)
        resume_data = parse_resume(resume_text)
    
    st.success("✅ Resume uploaded successfully!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Skills Detected", len(resume_data['skills']))
    with col2:
        st.metric("Years of Experience", resume_data['years_experience'])
    with col3:
        st.metric("Qualifications", len(resume_data['education']))
    
    st.subheader("🎯 Your Skills")
    skill_cols = st.columns(4)
    for idx, skill in enumerate(resume_data['skills']):
        with skill_cols[idx % 4]:
            st.write(f"✅ {skill}")
    
    st.markdown("---")
    st.subheader("💼 Recommended Jobs")
    
    jobs = get_sample_jobs(job_titles, locations)
    
    if jobs:
        jobs_with_scores = []
        for job in jobs:
            score = calculate_match_score(resume_data, job)
            jobs_with_scores.append({**job, 'match_score': score})
        
        jobs_with_scores = sorted(jobs_with_scores, key=lambda x: x['match_score'], reverse=True)
        
        for idx, job in enumerate(jobs_with_scores[:5], 1):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"""
                <div class="job-card">
                    <h4>{idx}. {job['title']}</h4>
                    <p><strong>{job['company']}</strong> • {job['location']} • {job['salary']}</p>
                    <p>{job['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                score = job['match_score']
                if score >= 80:
                    color = "🟢"
                elif score >= 60:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"""
                <div style="text-align: center;">
                    <h3>{color}</h3>
                    <div class="match-score">{int(score)}%</div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👈 Upload your resume from the sidebar to get started!")
    st.markdown("""
    ## How It Works
    1. Upload Resume (PDF or TXT)
    2. AI extracts your skills
    3. Matches you to jobs
    4. See match scores
    5. Get recommendations
    """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Smart Job Matcher v1.0 | Built with Python & Streamlit</p>", unsafe_allow_html=True)