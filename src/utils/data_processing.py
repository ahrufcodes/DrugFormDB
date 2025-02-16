"""
DrugFormDB Data Processing Utilities
--------------------------------

This module provides utility functions for data processing and management.
It includes functions for:
- Data cleaning and validation
- File management
- Progress tracking
- Error handling

Author: Ahmad Rufai Yusuf
License: MIT
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Utility class for data processing and management.
    
    This class provides methods for handling data files, cleaning
    classifications, and managing the dataset workflow.
    """
    
    @staticmethod
    def load_classifications(
        file_path: Union[str, Path],
        clean_only: bool = False
    ) -> Dict[str, List[str]]:
        """
        Load drug classifications from file.
        
        Args:
            file_path: Path to classification file
            clean_only: Whether to filter out unknown/error cases
            
        Returns:
            Dict[str, List[str]]: Drug classifications
        """
        try:
            with open(file_path, 'r') as f:
                classifications = json.load(f)
            
            if clean_only:
                return {
                    drug: forms for drug, forms in classifications.items()
                    if forms != ["Unknown"] and forms != ["Error"]
                }
            
            return classifications
            
        except Exception as e:
            logger.error(f"Error loading classifications: {str(e)}")
            return {}
    
    @staticmethod
    def save_classifications(
        classifications: Dict[str, List[str]],
        file_path: Union[str, Path]
    ) -> None:
        """
        Save drug classifications to file.
        
        Args:
            classifications: Drug classifications to save
            file_path: Path to save file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(classifications, f, indent=2)
            logger.info(f"Saved classifications to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving classifications: {str(e)}")
    
    @staticmethod
    def split_classifications(
        classifications: Dict[str, List[str]]
    ) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Split classifications into clean and unknown sets.
        
        Args:
            classifications: Drug classifications to split
            
        Returns:
            Tuple of (clean_classifications, unknown_classifications)
        """
        clean = {}
        unknown = {}
        
        for drug, forms in classifications.items():
            if forms == ["Unknown"] or forms == ["Error"]:
                unknown[drug] = forms
            else:
                clean[drug] = forms
        
        return clean, unknown
    
    @staticmethod
    def merge_validation_results(
        classifications: Dict[str, List[str]],
        validation_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge classifications with validation results.
        
        Args:
            classifications: Drug classifications
            validation_df: Validation results DataFrame
            
        Returns:
            pd.DataFrame: Merged results
        """
        # Convert classifications to DataFrame
        class_df = pd.DataFrame([
            {"drug_name": drug, "forms": forms}
            for drug, forms in classifications.items()
        ])
        
        # Merge with validation results
        merged = pd.merge(
            class_df,
            validation_df,
            on="drug_name",
            how="left"
        )
        
        return merged
    
    @staticmethod
    def generate_dataset_stats(df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive dataset statistics.
        
        Args:
            df: Dataset DataFrame
            
        Returns:
            Dict: Dataset statistics
        """
        total = len(df)
        
        # Confidence level stats
        high_conf = df[df["similarity_score"] >= 0.92]
        med_conf = df[(df["similarity_score"] >= 0.85) & (df["similarity_score"] < 0.92)]
        low_conf = df[df["similarity_score"] < 0.85]
        
        # Form distribution
        form_dist = df["gpt4_form"].value_counts()
        
        return {
            "total_drugs": total,
            "confidence_levels": {
                "high": {
                    "count": len(high_conf),
                    "percentage": len(high_conf) / total * 100,
                    "accuracy": high_conf["agrees_with_gpt4"].mean() * 100
                },
                "medium": {
                    "count": len(med_conf),
                    "percentage": len(med_conf) / total * 100,
                    "accuracy": med_conf["agrees_with_gpt4"].mean() * 100
                },
                "low": {
                    "count": len(low_conf),
                    "percentage": len(low_conf) / total * 100,
                    "accuracy": low_conf["agrees_with_gpt4"].mean() * 100
                }
            },
            "form_distribution": form_dist.to_dict(),
            "agreement_rate": df["agrees_with_gpt4"].mean() * 100,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    """Main entry point for command-line usage."""
    # Example usage
    processor = DataProcessor()
    
    # Load and process classifications
    classifications = processor.load_classifications(
        "../data/gpt4_classifications.json"
    )
    
    # Split into clean and unknown
    clean, unknown = processor.split_classifications(classifications)
    
    # Save split files
    processor.save_classifications(clean, "../data/clean_classifications.json")
    processor.save_classifications(unknown, "../data/unknown_classifications.json")
    
    # Load validation results
    validation_df = pd.read_csv("../data/validation_summary.csv")
    
    # Generate statistics
    stats = processor.generate_dataset_stats(validation_df)
    
    # Print statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total drugs: {stats['total_drugs']}")
    logger.info("\nConfidence Levels:")
    for level, data in stats['confidence_levels'].items():
        logger.info(f"\n{level.title()} Confidence:")
        logger.info(f"- Count: {data['count']}")
        logger.info(f"- Percentage: {data['percentage']:.1f}%")
        logger.info(f"- Accuracy: {data['accuracy']:.1f}%")
    
    logger.info(f"\nOverall Agreement Rate: {stats['agreement_rate']:.1f}%")

if __name__ == "__main__":
    main() 