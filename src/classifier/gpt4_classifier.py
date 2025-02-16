import json
import requests
import time
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")

client = OpenAI(api_key=api_key)

VALID_FORMS = [
    "Tablet", "Capsule", "Oral Solution", "Oral Suspension/Syrup",
    "Injection/Infusion", "Cream", "Ointment", "Gel", "Patch",
    "Inhaler/Nasal Spray", "Eye/Ear Drops", "Suppository"
]

def classify_drug(drug_name: str) -> Dict:
    """Classify a drug using GPT-4 API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a pharmaceutical expert with deep knowledge of drug formulations and delivery methods.
Consider the drug's active ingredients, therapeutic use, and typical administration routes.

Output format: Form1, Form2 (if multiple forms exist)
Only use these exact forms:
- Tablet
- Capsule
- Oral Solution
- Oral Suspension/Syrup
- Injection/Infusion
- Cream
- Ointment
- Gel
- Patch
- Inhaler/Nasal Spray
- Eye/Ear Drops
- Suppository

Examples:
- Amoxicillin -> Tablet, Capsule, Oral Suspension/Syrup
- Insulin -> Injection/Infusion
- Albuterol -> Inhaler/Nasal Spray
- Morphine -> Injection/Infusion, Tablet

If you're not completely certain about the form, respond with 'Unknown'."""
                },
                {
                    "role": "user",
                    "content": f"What are the administration forms for {drug_name}? Consider its therapeutic use and typical administration routes."
                }
            ],
            max_tokens=50,  # Increased for better context handling
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up and split multiple forms
        forms = [form.strip() for form in result.split(',')]
        
        # Validate each form
        valid_forms = [form for form in forms if form in VALID_FORMS]
        
        return {
            "forms": valid_forms if valid_forms else ["Unknown"]
        }
            
    except Exception as e:
        print(f"Error classifying {drug_name}: {str(e)}")
        return {"forms": ["Error"]}

def main():
    # Load drug names
    print("Loading drug names...")
    with open('../data/approved_drug_names.json', 'r') as f:
        data = json.load(f)
        drug_names = set(data['approved_drugs'])  # Convert to set for faster lookup

    print(f"Total number of drugs: {len(drug_names)}")
    
    # Load existing classifications if they exist
    classifications = {}
    if os.path.exists('../data/gpt4_classifications.json'):
        print("\nLoading existing progress...")
        with open('../data/gpt4_classifications.json', 'r') as f:
            classifications = json.load(f)
        print(f"Loaded {len(classifications)} existing classifications")
    
    # Find drugs to process (missing or errored)
    drugs_to_process = []
    for drug in drug_names:
        if drug not in classifications or classifications[drug] == ["Error"]:
            drugs_to_process.append(drug)
    
    print(f"\nDrugs to process:")
    print(f"- Missing drugs: {len([d for d in drugs_to_process if d not in classifications])}")
    print(f"- Error drugs: {len([d for d in drugs_to_process if d in classifications and classifications[d] == ['Error']])}")
    print(f"Total to process: {len(drugs_to_process)}")
    
    if not drugs_to_process:
        print("All drugs have been processed successfully!")
        return
    
    # Process drugs
    print("\nProcessing drugs...")
    BATCH_SIZE = 100  # Reduced batch size for better progress tracking
    
    for i in range(0, len(drugs_to_process), BATCH_SIZE):
        batch = drugs_to_process[i:i + BATCH_SIZE]
        print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(len(drugs_to_process) + BATCH_SIZE - 1)//BATCH_SIZE}...")
        
        for drug in batch:
            status = "New" if drug not in classifications else "Retry"
            print(f"Processing {drug} ({status})...")
            
            result = classify_drug(drug)
            classifications[drug] = result["forms"]
            print(f"Result: {result['forms']}")
            
            # Save after each drug to prevent loss of progress
            with open('../data/gpt4_classifications.json', 'w') as f:
                json.dump(classifications, f, indent=2)
            
            time.sleep(0.1)  # Rate limiting
        
        successful = len([d for d in classifications.values() if d != ["Error"]])
        print(f"\nProgress: {successful}/{len(drug_names)} drugs classified successfully ({(successful/len(drug_names)*100):.1f}%)")
        
        # Optional: Ask to continue after each batch
        if (i + BATCH_SIZE) < len(drugs_to_process):
            response = input("\nContinue to next batch? (yes/no): ")
            if response.lower() != 'yes':
                print("Saving progress and exiting...")
                break
    
    # Final statistics
    final_stats = {
        "total": len(drug_names),
        "processed": len(classifications),
        "successful": len([d for d in classifications.values() if d != ["Error"]]),
        "errors": len([d for d in classifications.values() if d == ["Error"]]),
        "unknown": len([d for d in classifications.values() if d == ["Unknown"]])
    }
    
    print("\nClassification complete!")
    print(f"Final statistics:")
    print(f"- Total drugs: {final_stats['total']}")
    print(f"- Successfully classified: {final_stats['successful']}")
    print(f"- Errors: {final_stats['errors']}")
    print(f"- Unknown: {final_stats['unknown']}")
    print(f"\nResults saved to gpt4_classifications.json")

if __name__ == "__main__":
    main()