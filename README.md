# DrugFormDB: Medication Form Classification Dataset 🏥

[![HuggingFace](https://img.shields.io/badge/🤗%20Dataset-DrugFormDB-yellow)](https://huggingface.co/datasets/ahruf/DrugFormDB)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Introduction

DrugFormDB is a pioneering that works by combining the reasoning capabilities of GPT-4 with the domain-specific knowledge of PubMedBERT.

We've created a robust and reliable classification system for pharmaceutical forms.

### Why DrugFormDB?

In healthcare settings, accurate medication form information is crucial for:
- 🏥 Patient safety and proper drug administration
- 💊 Pharmacy inventory management
- 📱 Healthcare application development
- 🔬 Medical research and data analysis

Our system achieves this with a unique two-stage approach:
1. Initial classification using GPT-4's reasoning
2. Validation through PubMedBERT's medical knowledge

## 🌟 Key Features

- **3,150+ Drug Classifications**: Comprehensive coverage of approved medications
- **Two-Stage Validation**: GPT-4 classification + PubMedBERT validation
- **Confidence Scoring**: Three-tier confidence system (93.0% accuracy for high confidence)
- **12 Standardized Forms**: From tablets to injections, covering all major drug forms

### Confidence Levels

| Level | Score | Accuracy | Use Case |
|-------|--------|----------|-----------|
| High | ≥0.92 | 93.0% | Production-ready classifications |
| Medium | 0.85-0.91 | 67.5% | Requires minimal verification |
| Low | <0.85 | - | Needs human review |

## 📊 Dataset Access

The complete dataset is hosted on Hugging Face, making it easily accessible for research and development:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ahruf/DrugFormDB")

# Get high-confidence classifications
high_confidence = dataset.filter(lambda x: x["confidence_level"] == "high")

# Look up specific drugs
amoxicillin = dataset.filter(lambda x: x["drug_name"] == "Amoxicillin")
```

### Dataset Format

Each record contains:
```python
{
    "drug_name": str,           # Name of the medication
    "gpt4_forms": List[str],    # Forms suggested by GPT-4
    "confidence_level": str,    # "high", "medium", or "low"
    "similarity_score": float,  # PubMedBERT validation score
    "agrees_with_gpt4": bool   # Cross-model agreement
}
```

## 📁 Repository Structure

```
.
├── src/
│   ├── classifier/           # Classification system
│   │   ├── gpt4_classifier.py
│   │   └── validator.py
│   ├── analysis/            # Analysis tools
│   │   └── visualization.py
│   └── utils/               # Utility functions
├── docs/                    # Documentation
└── requirements.txt         # Project dependencies
```

## 🛠️ Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Marmar-org/DrugFormDB.git
cd DrugFormDB
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📈 Visualizations & Analysis

Our comprehensive analysis provides insights into the classification system's performance:

### Confidence-Accuracy Relationship
![Confidence Accuracy](https://huggingface.co/datasets/ahruf/DrugFormDB/resolve/main/visualizations/confidence_accuracy.png)
*Shows the strong correlation between confidence scores and classification accuracy*

### Form Distribution
![Form Distribution](https://huggingface.co/datasets/ahruf/DrugFormDB/resolve/main/visualizations/form_distribution.png)
*Illustrates the distribution of different drug forms in our dataset*

### Cross-Form Similarity
![Similarity Heatmap](https://huggingface.co/datasets/ahruf/DrugFormDB/resolve/main/visualizations/similarity_heatmap.png)
*Visualizes the semantic relationships between different drug forms*

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Code Contributions**
   - Fork the repository
   - Create a feature branch
   - Submit a pull request

2. **Dataset Improvements**
   - Add new drug classifications
   - Validate existing classifications
   - Report inconsistencies

3. **Documentation**
   - Improve technical documentation
   - Add usage examples
   - Fix typos or unclear sections

For major changes, please open an issue first to discuss what you would like to change.

## 📝 Citation

If you use this dataset in your research, please cite:
```bibtex
@misc{drugformdb2024,
  title={DrugFormDB: A Dataset for Medication Form Classification},
  author={Ahmad Rufai Yusuf},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/ahruf/DrugFormDB}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ahmad Rufai Yusuf**
- Twitter: [@ahruf](https://x.com/ahruf)
- Blog: [ahruf.substack.com](https://ahruf.substack.com)
- LinkedIn: [Ahmad Rufai Yusuf](https://linkedin.com/in/ahmadrufai)

## 🙏 Acknowledgments

- OpenAI's GPT-4 team for their powerful language model
- Microsoft Research for PubMedBERT
- The open-source medical informatics community
- All contributors and users of DrugFormDB

## 📚 Further Reading

For more detailed information about the project:
- [Technical Documentation](docs/technical_documentation.md)
- [Validation Process](docs/validation_process_explained.md)
- [Blog Post](https://ahruf.substack.com) about the development process 