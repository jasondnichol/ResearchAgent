"""Test Claude API connection"""
import os
import anthropic
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    return os.getenv("CLAUDE_API_KEY")

# Test the API
def test_api():
    print("Testing Claude API connection...")
    
    client = anthropic.Anthropic(api_key=get_api_key())
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Respond with exactly: 'API connection successful!'"}
        ]
    )
    
    print(f"✅ {message.content[0].text}")
    print(f"📊 Tokens used: {message.usage.input_tokens} in, {message.usage.output_tokens} out")

if __name__ == "__main__":
    test_api()