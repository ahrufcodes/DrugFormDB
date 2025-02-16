from sentence_transformers import SentenceTransformer
import json
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm

def load_model():
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    return model

def create_reference_embeddings(model):
    reference_pairs = {
        "Tablet": [
            "Amoxicillin is taken as a tablet",
            "Aspirin is administered as a tablet",
            "Lisinopril comes in tablet form"
        ],
        "Capsule": [
            "Omeprazole is taken as a capsule",
            "Doxycycline is administered as a capsule",
            "Fluoxetine comes in capsule form"
        ],
        "Injection/Infusion": [
            "Insulin is administered via injection",
            "Heparin is given through injection",
            "Morphine can be administered through injection"
        ],
        "Cream": [
            "Hydrocortisone is applied as a cream",
            "Clotrimazole is used as a cream",
            "Betamethasone is administered as a cream"
        ],
        "Inhaler/Nasal Spray": [
            "Albuterol is administered through an inhaler",
            "Fluticasone is used as a nasal spray",
            "Salbutamol is taken via inhaler"
        ],
        "Eye/Ear Drops": [
            "Timolol is administered as eye drops",
            "Ciprofloxacin can be given as ear drops",
            "Latanoprost is used as eye drops"
        ]
    }
    
    reference_embeddings = {}
    print("Creating reference embeddings...")
    for form, statements in reference_pairs.items():
        embeddings = model.encode(statements)
        reference_embeddings[form] = np.mean(embeddings, axis=0)
    
    return reference_embeddings

def validate_drug_form(model, reference_embeddings, drug_name, form):
    test_statement = f"{drug_name} is administered as a {form}"
    test_embedding = model.encode(test_statement)
    
    similarities = {}
    for ref_form, ref_embedding in reference_embeddings.items():
        similarity = np.dot(test_embedding, ref_embedding) / (np.linalg.norm(test_embedding) * np.linalg.norm(ref_embedding))
        similarities[ref_form] = float(similarity)  # Convert to float for JSON serialization
    
    claimed_similarity = similarities.get(form, 0)
    best_form = max(similarities.items(), key=lambda x: x[1])
    
    return {
        "claimed_form": form,
        "claimed_similarity": claimed_similarity,
        "best_matching_form": best_form[0],
        "best_matching_score": best_form[1],
        "all_similarities": similarities,
        "agrees_with_gpt4": form == best_form[0]
    }

def main():
    # Load GPT-4 classifications
    print("Loading GPT-4 classifications...")
    with open('../data/gpt4_classifications.json', 'r') as f:
        gpt4_classifications = json.load(f)
    
    # Load model and create reference embeddings
    model = load_model()
    reference_embeddings = create_reference_embeddings(model)
    
    # Validate all drugs
    results = {}
    print("\nValidating all drugs...")
    for drug, forms in tqdm(gpt4_classifications.items()):
        if forms != ["Unknown"] and forms != ["Error"]:
            # For now, we'll just validate the first form if multiple exist
            validation = validate_drug_form(model, reference_embeddings, drug, forms[0])
            results[drug] = validation
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'../analysis/validation_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame for analysis
    summary_data = []
    for drug, result in results.items():
        row = {
            'drug': drug,
            'gpt4_form': result['claimed_form'],
            'best_match': result['best_matching_form'],
            'similarity_score': result['claimed_similarity'],
            'best_match_score': result['best_matching_score'],
            'agrees_with_gpt4': result['agrees_with_gpt4']
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Save summary stats
    print("\nSummary Statistics:")
    print(f"Total drugs validated: {len(results)}")
    print(f"GPT-4 and Model agree on: {df['agrees_with_gpt4'].sum()} drugs ({(df['agrees_with_gpt4'].mean()*100):.1f}%)")
    
    # Save summary to CSV
    csv_file = f'../data/validation_summary_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Summary saved to: {csv_file}")
    
    # Print form-wise agreement
    print("\nAgreement by form:")
    form_agreement = df[df['agrees_with_gpt4']]['gpt4_form'].value_counts()
    form_total = df['gpt4_form'].value_counts()
    for form in form_total.index:
        agree = form_agreement.get(form, 0)
        total = form_total[form]
        print(f"{form}: {agree}/{total} ({(agree/total*100):.1f}%)")

if __name__ == "__main__":
    main() 