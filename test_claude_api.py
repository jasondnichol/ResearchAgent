"""Test Claude API connection"""
import anthropic
import subprocess

# Get API key from Windows Credential Manager
def get_api_key():
    result = subprocess.run(
        ['cmdkey', '/list'],
        capture_output=True,
        text=True
    )
    # For now, just use the key directly (we'll improve this)
    return "***REDACTED***"

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