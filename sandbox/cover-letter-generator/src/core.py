from typing import List, Optional, BinaryIO
from pydantic import BaseModel, Field
import tempfile
import os
import asyncio
import logging
from openai import AsyncOpenAI
from firecrawl import FirecrawlApp
import PyPDF2
from .scraper import scrape_job_posting
import requests
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# Data Models with Pydantic
# --------------------------------------------------------------


class ResumeExtraction(BaseModel):
    experience: List[str] = Field(description="List of work experiences")
    skills: List[str] = Field(description="List of skills")
    education: List[str] = Field(description="List of education details")
    contact_info: str = Field(description="Contact information")


class JobExtraction(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    requirements: List[str] = Field(description="Job requirements")
    description: str = Field(description="Job description")


class CoverLetter(BaseModel):
    content: str = Field(description="Generated cover letter text")


# --------------------------------------------------------------
# Step 2. Create 3 functions
# --------------------------------------------------------------


async def extract_resume_info(
    client: AsyncOpenAI, pdf_text: str
) -> Optional[ResumeExtraction]:
    """First LLM call: Extract structured information from resume"""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a resume parser. Return ONLY a JSON object with this structure:
                    {
                        "experience": ["list of work experiences"],
                        "skills": ["list of technical and soft skills"],
                        "education": ["list of education details"],
                        "contact_info": "full contact information"
                    }
                    IMPORTANT: Return ONLY valid JSON, no other text.""",
                },
                {"role": "user", "content": pdf_text},
            ],
        )
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"Resume LLM Response: {response_text}")
        return ResumeExtraction.model_validate_json(response_text)
    except Exception as e:
        logger.error(f"Resume extraction failed: {str(e)}")
        logger.error(
            f"Response was: {response_text if 'response_text' in locals() else 'No response'}"
        )
        return None


async def extract_job_info(
    client: AsyncOpenAI, job_content: str
) -> Optional[JobExtraction]:
    """Second LLM call: Extract structured information from job posting"""
    try:
        # Convert job_content to string and handle potential None
        job_text = str(job_content) if job_content is not None else ""
        logger.info(f"Job content type: {type(job_text)}")

        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a job posting parser. Return ONLY a JSON object with this structure:
                    {
                        "title": "exact job title",
                        "company": "company name",
                        "requirements": ["list of key requirements"],
                        "description": "brief job description"
                    }
                    IMPORTANT: Return ONLY valid JSON, no other text.""",
                },
                {"role": "user", "content": job_text},
            ],
        )
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"Job LLM Response: {response_text}")
        return JobExtraction.model_validate_json(response_text)
    except Exception as e:
        logger.error(f"Job info extraction failed: {str(e)}")
        logger.error(
            f"Job content was: {job_text if 'job_text' in locals() else 'No content'}"
        )
        return None


async def generate_cover_letter(
    client: AsyncOpenAI, resume_info: ResumeExtraction, job_info: JobExtraction
) -> Optional[str]:
    """Third LLM call: Generate the cover letter using the results from previous async calls"""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """Write a compelling cover letter following these guidelines:
                    1. Start with a strong hook about the company/role
                    2. Focus on relevant achievements matching job requirements
                    3. Use specific metrics from past experience
                    4. Keep it concise (300-400 words)
                    5. End with a confident call to action""",
                },
                {
                    "role": "user",
                    "content": f"Resume: {resume_info.model_dump()}\nJob: {job_info.model_dump()}",
                },
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Cover letter generation failed: {str(e)}")
        return None


# --------------------------------------------------------------
# Main Processing Function
# --------------------------------------------------------------


async def process_cover_letter_request(
    resume_file: BinaryIO,
    job_url: str,
    api_key: str,
) -> str:
    """
    Process a cover letter generation request using Deepseek API.
    
    Args:
        resume_file: Uploaded resume file (PDF)
        job_url: URL of the job posting
        api_key: Deepseek API key
    
    Returns:
        str: Generated cover letter
    """
    try:
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(resume_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        
        logger.info(f"Extracted PDF text length: {len(resume_text)}")
        
        # Scrape job posting
        job_data = scrape_job_posting(job_url)
        job_description = job_data['content']
        
        # Prepare the API request to Deepseek
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": """You are a professional cover letter writer. 
                Your task is to create a compelling, personalized cover letter that matches 
                the candidate's experience with the job requirements. The letter should be 
                professional, engaging, and highlight relevant skills and experiences."""},
                {"role": "user", "content": f"""
                Please write a cover letter based on the following:
                
                Resume:
                {resume_text[:2000]}  # Limiting resume text to avoid token limits
                
                Job Description:
                {job_description[:2000]}  # Limiting job description to avoid token limits
                
                Make sure to:
                1. Match key skills and experiences from the resume to job requirements
                2. Use a professional tone
                3. Keep it concise (about 300-400 words)
                4. Include a strong opening and closing
                5. Format it as a proper business letter
                """}
            ],
            "temperature": 0.7
        }
        
        # Make the API request to Deepseek
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        return result['choices'][0]['message']['content']
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise
