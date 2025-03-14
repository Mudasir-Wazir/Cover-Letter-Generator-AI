from dotenv import load_dotenv
import os

def test_env_variables():
    # Load environment variables
    load_dotenv()
    
    # Get environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    
    # Check OpenAI API key
    print("\nTesting OpenAI API key:")
    if openai_key:
        print("✅ OpenAI API key is set")
        print(f"   Length: {len(openai_key)} characters")
    else:
        print("❌ OpenAI API key is not set")
    
    # Check Firecrawl API key
    print("\nTesting Firecrawl API key:")
    if firecrawl_key:
        print("✅ Firecrawl API key is set")
        print(f"   Length: {len(firecrawl_key)} characters")
    else:
        print("❌ Firecrawl API key is not set")

if __name__ == "__main__":
    test_env_variables() 