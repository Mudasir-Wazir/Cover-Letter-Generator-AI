import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def scrape_job_posting(url: str) -> dict:
    """
    Scrape job posting details from a given URL.
    
    Args:
        url (str): The URL of the job posting
        
    Returns:
        dict: Dictionary containing job details
    """
    try:
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract text content
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            'url': url,
            'content': text,
            'title': soup.title.string if soup.title else '',
        }
        
    except requests.RequestException as e:
        logger.error(f"Error scraping job URL: {str(e)}")
        return {
            'url': url,
            'content': f"Error scraping content: {str(e)}",
            'title': 'Error'
        }
    except Exception as e:
        logger.error(f"Unexpected error while scraping: {str(e)}")
        return {
            'url': url,
            'content': f"Unexpected error: {str(e)}",
            'title': 'Error'
        } 