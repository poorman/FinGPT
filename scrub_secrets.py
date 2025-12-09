import json
import os

token_to_remove = "REDACTED_HF_TOKEN"
replacement_token = "YOUR_HF_TOKEN"

files_to_scrub = [
    "/root/projects/FinGPT/fingpt/FinGPT_MultiAgentsRAG/Fine_tune_model/fine_tune_GLM2.ipynb",
    "/root/projects/FinGPT/fingpt/FinGPT_MultiAgentsRAG/Fine_tune_model/fine_tune_Llama2.ipynb"
]

def scrub_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if token_to_remove in content:
            print(f"Found token in {filepath}. Scrubbing...")
            new_content = content.replace(token_to_remove, replacement_token)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Successfully scrubbed {filepath}")
        else:
            print(f"Token not found in {filepath}")
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    for filepath in files_to_scrub:
        scrub_file(filepath)
