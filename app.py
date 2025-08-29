from __future__ import annotations
import os
import io
import time
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
import pdfplumber

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, NotFound, GoogleAPICallError

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(page_title="TalentFlow", layout="wide")
st.title("TalentFlow — AI ATS")
st.caption("“Streamline Your Hiring with AI”")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def smart_generate(parts: list, temperature: float = 0.2, max_retries: int = 1) -> str:
    if not API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY. Add it to your .env or Streamlit secrets.")

    for attempt in range(max_retries + 1):
        try:
            resp = genai.GenerativeModel("gemini-1.5-pro").generate_content(parts, generation_config={"temperature": temperature})
            return resp.text
        except (ResourceExhausted, NotFound, GoogleAPICallError):
            break

    for attempt in range(max_retries + 1):
        try:
            resp = genai.GenerativeModel("gemini-1.5-flash").generate_content(parts, generation_config={"temperature": temperature})
            return resp.text
        except GoogleAPICallError as e:
            if attempt == max_retries:
                raise
            time.sleep(1.5 * (attempt + 1))

# -----------------------------
# Prompts
# -----------------------------
def review_prompt(jd: str, resume: str) -> str:
    return f"""
You are an experienced HR manager. Review the resume below against the job description.
Job Description: {jd}
Resume: {resume}
Provide strengths, weaknesses, and final thoughts in a clear text format.
"""

def advice_prompt(jd: str, resume: str) -> str:
    return f"""
You are an experienced HR manager. Based on the resume and job description, provide personalized skill improvement advice in simple readable text.
Job Description: {jd}
Resume: {resume}
"""

def match_prompt(jd: str, resume: str) -> str:
    return f"""
You are an ATS scanner. Compare the resume against the job description.
Job Description: {jd}
Resume: {resume}
Provide a readable output with percentage match, missing keywords, matched keywords, and final thoughts.
"""

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
temp = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
max_resume_chars = st.sidebar.number_input("Max resume characters sent", min_value=2000, max_value=50000, value=15000, step=1000)

# -----------------------------
# Inputs
# -----------------------------
jd_text = st.text_area("Job Description", placeholder="Paste the JD here...", height=220)
up = st.file_uploader("Upload Resume (PDF)", type=["pdf"], accept_multiple_files=False)

resume_text = ""
if up is not None:
    file_bytes = up.read()
    with st.spinner("Parsing PDF..."):
        try:
            resume_text = extract_pdf(file_bytes)
            st.success("📄 Resume uploaded successfully!")
        except Exception as e:
            st.error(f"Failed to parse PDF: {e}")

analyze = st.button("Analyze Your Resume")

# -----------------------------
# Run Analysis
# -----------------------------
if analyze:
    if not jd_text:
        st.warning("Please paste a Job Description.")
    if not resume_text:
        st.warning("Please upload a valid PDF resume.")

    if jd_text and resume_text:
        trimmed_resume = resume_text[: int(max_resume_chars)]
        parts_base = []
        tab1, tab2, tab3 = st.tabs(["Resume Review", "Skill Advice", "ATS Match %"])

        with tab1:
            with st.spinner("Reviewing resume vs JD..."):
                text = smart_generate(parts_base + [review_prompt(jd_text, trimmed_resume)], temperature=temp)
                st.subheader("Review")
                st.write(text)

        with tab2:
            with st.spinner("Generating skill advice..."):
                text = smart_generate(parts_base + [advice_prompt(jd_text, trimmed_resume)], temperature=temp)
                st.subheader("Skill Advice")
                st.markdown(f'<div style="white-space: pre-wrap;">{text}</div>', unsafe_allow_html=True)

        with tab3:
            with st.spinner("Calculating ATS match..."):
                text = smart_generate(parts_base + [match_prompt(jd_text, trimmed_resume)], temperature=temp)
                st.subheader("ATS Match")
                # Color-code percentage if present in output
                match_percent = 0
                import re
                m = re.search(r'(\d+)%', text)
                if m:
                    match_percent = int(m.group(1))
                    color = "red" if match_percent < 50 else "orange" if match_percent < 80 else "green"
                    st.markdown(f'<span style="color:{color}; font-weight:bold; font-size:18px;">Match: {match_percent}%</span>', unsafe_allow_html=True)
                    text = re.sub(r'\d+%', '', text)
                st.markdown(f'<div style="white-space: pre-wrap;">{text.strip()}</div>', unsafe_allow_html=True)
