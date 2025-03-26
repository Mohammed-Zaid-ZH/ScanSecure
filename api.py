import requests
#import Bigram_model as bi
# Replace with your actual API key
API_KEY = "AIzaSyAAKt_LSFJQQQ1SZ6GDlLBOuy3KizCNqAE"
SAFE_BROWSING_URL = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

def check_url_google_safe(url):
    payload = {
        "client": {
            "clientId": "your-client-id",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    params = {"key": API_KEY}
    response = requests.post(SAFE_BROWSING_URL, json=payload, params=params)
    result = response.json()
    
    if "matches" in result:
        return "Unsafe URL detected by Google Safe Browsing!"
    else:
        return "safe"
        

# Example usage
url_to_check = "http://malware.testing.google.test/testing/malware/"
print(check_url_google_safe(url_to_check))