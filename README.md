# RAGChecker

## About

RAGChecker is an open-source visual analytics tool for detecting hallucinations in Large Language Model (LLM) outputs. It helps users identify when an LLM generates convincing but inaccurate information by comparing the model's responses against source documents.

## The Problem

When LLMs are enhanced with closed-source knowledge (like internal documents or proprietary data) to answer questions, they sometimes produce hallucinations—confident yet factually incorrect statements. These errors are often convincing enough to escape detection by human reviewers, making automated detection crucial.

## How It Works

RAGChecker extracts structured knowledge representations from both source documents and LLM outputs, then compares them to identify discrepancies. The tool offers two detection methods:

### 1. GraphEval+
- Extracts subject-verb-object triples using GPT-4o
- Provides comprehensive analysis with rich semantic relationships
- Processing time: ~8 hours (suitable for thorough analysis)

### 2. SICI (Sentence Isolation + Coreference Isolation)
- Analyzes individual sentences using spaCy for Named Entity Recognition
- Uses traditional NLP techniques for coreference resolution
- **70% detection accuracy** while maintaining computational efficiency
- Processing time: ~30 minutes (suitable for production use)
- Available in two variants:
  - **SICI-0**: Analyzes sentences in isolation
  - **SICI-1**: Includes adjacent sentences for better context

## Key Features

- **Interactive Visualization**: Quadrant-based graph showing claim reliability
  - X-axis: Factual consistency score (NLI)
  - Y-axis: Semantic similarity to source material
  - Color-coded nodes: Green (reliable) to Red (potential hallucination)

- **Claim Categorization**: Automatic grouping into four categories
  - 🟢 High Reliability
  - 🟡 Suspicious Content
  - 🟠 Plausible But Unsupported
  - 🔴 Potential Hallucination

- **User Feedback Loop**: Select problematic claims and generate improved responses

- **Source Tracing**: Visualize connections between LLM claims and source material

## Technical Details

RAGChecker uses:
- **Semantic Similarity**: OpenAI embeddings for consistent comparison across methods
- **Coreference Resolution**: spaCy for efficient entity resolution in SICI methods
- **Triple Extraction**: GPT-4o for GraphEval+ knowledge graphs
- **Visualization**: Matplotlib for interactive graph layouts

## Research Paper

This tool is based on research presented in:

**"Graphing the Truth: Structured Visualizations for Automated Hallucination Detection in LLMs"**  
by Tanmay Agrawal

[📄 Read the full paper](https://drive.google.com/file/d/1OIBSYVlYUdV2SrE5RIodxZGcZcnesRcI/view?usp=sharing)

## Installation

### Requirements

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### spaCy Model

For SICI methods, install the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

### Environment Variables

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run interactive_app.py
```

Then:
1. Select a detection method (GraphEval+ or SICI)
2. Provide source context and a question
3. Click "Run Analysis" to generate and validate the LLM response
4. Review the visualization and claim details
5. Select problematic claims and generate an improved response

## Performance Comparison

| Method | Accuracy | Processing Time | Best For |
|--------|----------|----------------|----------|
| GraphEval | 71.5% | N/A | Baseline comparison |
| GraphEval+ | 53.0% | ~8 hours | Comprehensive analysis |
| SICI-0 | 67.0% | ~30 minutes | Quick validation |
| SICI-1 | 70.0% | ~30 minutes | **Production use** ✓ |

*Evaluated on SummEval dataset*

## Use Cases

- **Content Verification**: Validate LLM-generated reports against source documents
- **RAG System Auditing**: Detect when retrieval-augmented generation produces hallucinations
- **Quality Assurance**: Review AI-generated content before publication
- **Research**: Analyze patterns in how LLMs transform source knowledge

## Citation

If you use RAGChecker in your research, please cite:
```bibtex
@article{agrawal2025graphing,
  title={Graphing the Truth: Structured Visualizations for Automated Hallucination Detection in LLMs},
  author={Agrawal, Tanmay},
  year={2025},
  institution={University of Arizona}
}
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

Tanmay Agrawal  
Email: tanmayagrawal21@arizona.edu

---

**Note**: This tool provides automated detection of potential hallucinations but should be used as part of a comprehensive verification workflow. Always review flagged content manually for critical applications.