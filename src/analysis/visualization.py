import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import numpy as np
from matplotlib.sankey import Sankey
from datetime import datetime

def load_latest_results():
    # Find the latest validation files
    json_files = list(Path('../analysis').glob('validation_results_*.json'))
    csv_files = list(Path('../data').glob('validation_summary_*.csv'))
    
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Loading results from: {latest_json.name}")
    
    with open(latest_json, 'r') as f:
        detailed_results = json.load(f)
    
    df = pd.read_csv(latest_csv)
    # Add timestamp column for temporal analysis
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df))
    return detailed_results, df

def plot_agreement_by_form(df):
    plt.figure(figsize=(15, 8))
    
    # Calculate agreement percentages
    form_agreement = df[df['agrees_with_gpt4']]['gpt4_form'].value_counts()
    form_total = df['gpt4_form'].value_counts()
    agreement_pct = (form_agreement / form_total * 100).round(1)
    
    # Sort by percentage
    agreement_pct = agreement_pct.sort_values(ascending=True)
    
    # Create bar plot
    bars = plt.barh(range(len(agreement_pct)), agreement_pct)
    plt.yticks(range(len(agreement_pct)), agreement_pct.index)
    
    # Add percentage labels
    for i, v in enumerate(agreement_pct):
        plt.text(v + 1, i, f'{v}%', va='center')
    
    plt.title('Agreement Percentage by Drug Form')
    plt.xlabel('Agreement Percentage')
    plt.ylabel('Drug Form')
    
    # Add total counts
    for i, (form, pct) in enumerate(agreement_pct.items()):
        total = form_total[form]
        agree = form_agreement.get(form, 0)
        plt.text(0, i, f' {agree}/{total}', va='center', ha='right', color='white')
    
    plt.tight_layout()
    plt.savefig('../analysis/agreement_by_form.png')
    plt.close()

def plot_similarity_distributions(df):
    plt.figure(figsize=(15, 8))
    
    # Create violin plot
    sns.violinplot(data=df, x='gpt4_form', y='similarity_score')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Similarity Scores by Drug Form')
    plt.xlabel('Drug Form')
    plt.ylabel('Similarity Score')
    
    plt.tight_layout()
    plt.savefig('../analysis/similarity_distributions.png')
    plt.close()

def analyze_disagreements(df, detailed_results):
    # Find cases where GPT-4 and model disagree
    disagreements = df[~df['agrees_with_gpt4']].copy()
    
    # Add similarity score differences
    disagreements['score_difference'] = disagreements['best_match_score'] - disagreements['similarity_score']
    
    # Sort by score difference to find most significant disagreements
    significant_disagreements = disagreements.sort_values('score_difference', ascending=False)
    
    # Save top 20 most significant disagreements
    print("\nTop 20 Most Significant Disagreements:")
    print("=======================================")
    
    with open('../analysis/significant_disagreements.txt', 'w') as f:
        f.write("Top 20 Most Significant Disagreements\n")
        f.write("=====================================\n\n")
        
        for _, row in significant_disagreements.head(20).iterrows():
            drug = row['drug']
            details = detailed_results[drug]
            
            output = f"Drug: {drug}\n"
            output += f"GPT-4 Classification: {row['gpt4_form']} (score: {row['similarity_score']:.3f})\n"
            output += f"Model Suggestion: {row['best_match']} (score: {row['best_match_score']:.3f})\n"
            output += "All form similarities:\n"
            
            # Sort similarities by score
            similarities = sorted(details['all_similarities'].items(), key=lambda x: x[1], reverse=True)
            for form, score in similarities:
                output += f"- {form}: {score:.3f}\n"
            
            output += "\n"
            print(output)
            f.write(output)

def plot_confusion_matrix(df):
    plt.figure(figsize=(12, 8))
    
    # Create confusion matrix
    forms = df['gpt4_form'].unique()
    matrix_df = pd.crosstab(df['gpt4_form'], df['best_match'])
    
    # Plot heatmap
    sns.heatmap(matrix_df, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Confusion Matrix: GPT-4 vs Model Classifications')
    plt.xlabel('Model Classification')
    plt.ylabel('GPT-4 Classification')
    
    plt.tight_layout()
    plt.savefig('../analysis/confusion_matrix.png')
    plt.close()

def plot_similarity_threshold_impact(df):
    plt.figure(figsize=(10, 6))
    
    # Calculate agreement rate at different similarity thresholds
    thresholds = np.arange(0.8, 1.0, 0.01)
    agreement_rates = []
    
    for threshold in thresholds:
        # Consider only high-confidence predictions
        high_confidence = df[df['similarity_score'] >= threshold]
        if len(high_confidence) > 0:
            agreement_rate = (high_confidence['agrees_with_gpt4'].mean() * 100)
            agreement_rates.append(agreement_rate)
        else:
            agreement_rates.append(0)
    
    plt.plot(thresholds, agreement_rates)
    plt.title('Agreement Rate vs Similarity Score Threshold')
    plt.xlabel('Similarity Score Threshold')
    plt.ylabel('Agreement Rate (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../analysis/threshold_impact.png')
    plt.close()

def plot_form_distribution(df, output_dir):
    plt.figure(figsize=(15, 10))
    # Create stacked bar chart showing successful vs failed classifications
    form_success = df[df['agrees_with_gpt4']].groupby('gpt4_form').size()
    form_failure = df[~df['agrees_with_gpt4']].groupby('gpt4_form').size()
    
    # Sort by total frequency
    total_freq = form_success + form_failure
    sort_order = total_freq.sort_values(ascending=True).index
    
    form_success = form_success.reindex(sort_order)
    form_failure = form_failure.reindex(sort_order)
    
    # Create bar plot
    plt.barh(range(len(form_success)), form_success, label='Successful', color='#2ecc71')
    plt.barh(range(len(form_failure)), form_failure, left=form_success, label='Failed', color='#e74c3c')
    
    plt.yticks(range(len(sort_order)), sort_order)
    plt.xlabel('Number of Classifications')
    plt.title('Distribution of Drug Forms and Classification Success')
    plt.legend(loc='lower right')
    
    # Add percentage labels
    for i, (success, failure) in enumerate(zip(form_success, form_failure)):
        total = success + failure
        pct = (success / total * 100) if total > 0 else 0
        plt.text(total + 1, i, f'{pct:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/form_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_similarity_heatmap(df, output_dir):
    plt.figure(figsize=(12, 12))
    # Create correlation matrix of similarity scores between forms
    similarity_matrix = pd.pivot_table(
        df,
        values='similarity_score',
        index='gpt4_form',
        columns='best_match',
        aggfunc='mean'
    )
    
    # Plot heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis',
                square=True, cbar_kws={'label': 'Mean Similarity Score'})
    plt.title('Cross-Form Similarity Scores')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_temporal_stability(df, output_dir):
    plt.figure(figsize=(15, 8))
    # Plot rolling average of classification accuracy
    rolling_accuracy = df.sort_values('timestamp')['agrees_with_gpt4'].rolling(100).mean()
    
    plt.plot(range(len(rolling_accuracy)), rolling_accuracy.values, 
            color='#3498db', linewidth=2)
    plt.fill_between(range(len(rolling_accuracy)), 
                    rolling_accuracy.values - rolling_accuracy.std(),
                    rolling_accuracy.values + rolling_accuracy.std(),
                    alpha=0.2, color='#3498db')
    
    plt.title('Classification Stability Over Time')
    plt.xlabel('Number of Classifications')
    plt.ylabel('100-drug Rolling Average Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_accuracy(df, output_dir):
    plt.figure(figsize=(10, 6))
    # Create bins of confidence scores
    df['confidence_bin'] = pd.qcut(df['similarity_score'], q=10)
    accuracy_by_confidence = df.groupby('confidence_bin')['agrees_with_gpt4'].mean()
    
    # Plot with enhanced styling
    plt.plot(range(len(accuracy_by_confidence)), accuracy_by_confidence.values * 100, 
            marker='o', color='#2ecc71', linewidth=2, markersize=8)
    
    # Add confidence threshold line
    plt.axvline(x=7, color='#e74c3c', linestyle='--', alpha=0.5, 
                label='0.92 Confidence Threshold')
    
    plt.title('Relationship between Confidence and Accuracy')
    plt.xlabel('Confidence Score Percentile')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_form_transitions(df, output_dir):
    plt.figure(figsize=(15, 10))
    # Calculate transitions for misclassifications
    transitions = df[~df['agrees_with_gpt4']].groupby(
        ['gpt4_form', 'best_match']
    ).size().reset_index(name='count')
    
    # Sort by frequency
    transitions = transitions.sort_values('count', ascending=False).head(10)
    
    # Create horizontal bar plot
    plt.barh(range(len(transitions)), transitions['count'], color='#3498db')
    plt.yticks(range(len(transitions)), 
              transitions['gpt4_form'] + ' → ' + transitions['best_match'])
    
    plt.title('Top 10 Most Common Form Transitions')
    plt.xlabel('Number of Occurrences')
    
    # Add count labels
    for i, v in enumerate(transitions['count']):
        plt.text(v + 0.5, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/form_transitions.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results
    detailed_results, df = load_latest_results()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'../analysis/analysis_output_{timestamp}'
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Generating enhanced visualizations...")
    
    # Generate all visualizations
    plot_form_distribution(df, output_dir)
    plot_similarity_heatmap(df, output_dir)
    plot_temporal_stability(df, output_dir)
    plot_confidence_accuracy(df, output_dir)
    plot_form_transitions(df, output_dir)
    
    print(f"\nAnalysis complete! Generated files in: {output_dir}")
    print("1. form_distribution.png - Distribution of drug forms and success rates")
    print("2. similarity_heatmap.png - Cross-form similarity score patterns")
    print("3. temporal_stability.png - Classification stability over time")
    print("4. confidence_accuracy.png - Relationship between confidence and accuracy")
    print("5. form_transitions.png - Most common misclassification patterns")

if __name__ == "__main__":
    main() 