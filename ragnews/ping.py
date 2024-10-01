import requests
import os

def ping_groq_api():
    url = "https://api.groq.com/openai/v1/chat/completions"  # Example endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"  # Securely access API key

    }
    
    data = {
        "model": "llama-3.1-8b-instant",  # Minimal model request (adjust as needed)
        "messages": [{"role": "system", "content": "ping"}]  # Minimal valid input
    }
    
    try:
        # Send a POST request to check API availability
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("API is available!")
        else:
            print(f"API is not available, status code: {response.status_code}")
            print(f"Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to the API: {e}")

if __name__ == "__main__":
    ping_groq_api()