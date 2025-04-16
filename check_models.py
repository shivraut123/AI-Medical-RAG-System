import google.generativeai as genai
import os

# PASTE YOUR KEY HERE
os.environ["GOOGLE_API_KEY"] = "AIzaSyD2e1Mwxi3hdF7DwmmKyMRUz9s_LgACxvo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Checking available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)