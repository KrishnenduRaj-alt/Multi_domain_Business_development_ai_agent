# test_key.py
import os
from dotenv import load_dotenv

print("--- Starting API Key Test ---")

# Step 1: Load the .env file
print("Attempting to load the .env file...")
load_dotenv()
print(".env file should be loaded now.")

# Step 2: Read the API key from the environment
print("Attempting to read the OPENAI_API_KEY...")
api_key = os.getenv("OPENAI_API_KEY")

# Step 3: Check and report the result
if api_key:
    # We will only show the first 5 and last 4 characters for security
    print("\nSUCCESS! The API key was found.")
    print(f"Your key starts with: {api_key[:5]}... and ends with: ...{api_key[-4:]}")
else:
    print("\nFAILURE! The API key was NOT found.")
    print("Please double-check that your .env file is in the correct folder, is named correctly, and the key name is OPENAI_API_KEY.")

print("--- Test Complete ---")