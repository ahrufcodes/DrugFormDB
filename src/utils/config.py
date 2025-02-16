"""
DrugFormDB Configuration Module
---------------------------

This module manages configuration settings and constants for the DrugFormDB project.
It includes:
- Path configurations
- Model parameters
- Validation thresholds
- Drug form definitions

Author: Ahmad Rufai Yusuf
License: MIT
"""

import os
from pathlib import Path
from typing import Dict, List, Set

# Project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# File paths
CLEAN_CLASSIFICATIONS_PATH = DATA_DIR / "clean_classifications.json"
UNKNOWN_CLASSIFICATIONS_PATH = DATA_DIR / "unknown_classifications.json"
VALIDATION_SUMMARY_PATH = DATA_DIR / "validation_summary.csv"

# Model parameters
GPT4_MODEL = "gpt-4-1106-preview"
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Validation thresholds
CONFIDENCE_THRESHOLDS = {
    "high": 0.92,
    "medium": 0.85,
    "low": 0.0
}

# Drug form definitions and examples
DRUG_FORMS: Dict[str, List[str]] = {
    "tablet": [
        "oral tablet",
        "solid oral dosage form",
        "compressed tablet",
        "film-coated tablet"
    ],
    "capsule": [
        "oral capsule",
        "hard gelatin capsule",
        "soft gelatin capsule",
        "extended-release capsule"
    ],
    "injection": [
        "injectable solution",
        "intravenous injection",
        "intramuscular injection",
        "subcutaneous injection"
    ],
    "oral_solution": [
        "oral liquid",
        "oral suspension",
        "oral syrup",
        "elixir"
    ],
    "topical": [
        "cream",
        "ointment",
        "gel",
        "lotion",
        "patch"
    ],
    "inhalation": [
        "inhaler",
        "nebulizer solution",
        "respiratory inhalant",
        "metered dose inhaler"
    ]
}

# Valid drug forms set for quick lookup
VALID_DRUG_FORMS: Set[str] = {
    form for forms in DRUG_FORMS.values() 
    for form in forms
}

# Visualization settings
VIZ_SETTINGS = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn",
    "palette": "viridis",
    "font_scale": 1.2
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# API configuration
API_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1,  # seconds
    "timeout": 30,     # seconds
    "batch_size": 50   # items per batch
}

def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return ROOT_DIR

def get_data_dir() -> Path:
    """Get the absolute path to the data directory."""
    return DATA_DIR

def get_visualizations_dir() -> Path:
    """Get the absolute path to the visualizations directory."""
    return VISUALIZATIONS_DIR

def is_valid_drug_form(form: str) -> bool:
    """Check if a given drug form is valid."""
    return form.lower() in VALID_DRUG_FORMS

def get_form_examples(form_type: str) -> List[str]:
    """Get examples for a specific drug form type."""
    return DRUG_FORMS.get(form_type, []) 