# Round 1B: Persona-Driven Document Intelligence

## Approach Overview

This solution implements a persona-driven document intelligence system that processes JSON outputs from Round 1A and ranks document sections based on their relevance to a specific persona and job-to-be-done. The system takes structured outline data and applies intelligent ranking to surface the most relevant sections.

### 1. JSON Input Processing
- **Round 1A Integration**: Reads JSON files containing document outlines with titles, headings (H1, H2, H3), and page numbers
- **Data Structure**: Converts JSON outline data into structured DocumentSection objects for processing
- **Document Mapping**: Maintains relationship between JSON files and their corresponding PDF document names

### 2. Relevance Scoring Algorithm
The system calculates relevance scores using multiple approaches:

- **Semantic Similarity**: When spaCy word vectors are available, uses cosine similarity between section titles and the persona+job query
- **Keyword Matching**: Extracts key terms from persona and job descriptions, scoring sections based on term matches in section titles
- **Hierarchical Weighting**: Provides bonus scoring for higher-level headings (H1 > H2 > H3) as they typically contain more important information
- **Domain Knowledge**: Incorporates common important section keywords (introduction, methodology, ingredients, etc.) for additional relevance scoring

### 3. Ranking and Output Generation
- Combines multiple scoring factors with weighted importance
- Ranks all sections by relevance score in descending order
- Generates top 10 sections for the extracted_sections list with importance ranking
- Creates detailed analysis for top 5 sections with contextual refined_text

## Models and Libraries Used

### Core Dependencies
- **spaCy**: Natural language processing and semantic similarity
  - Primary: `en_core_web_sm` (lightweight, ~15MB)
  - Provides tokenization, semantic analysis, and word similarity
- **NumPy**: Numerical computations for scoring algorithms
- **Standard Library**: JSON processing, file handling, command-line argument parsing

### Key Features
- **Lightweight Design**: Minimal dependencies focused on text analysis
- **Offline Operation**: All processing is local, no internet access required
- **Adaptive Intelligence**: Falls back gracefully when advanced NLP features aren't available
- **Performance Optimized**: Designed to meet the 60-second processing constraint

## Architecture Design

The system follows a clean, modular architecture:

```
PersonaDrivenDocumentIntelligence
├── process_document_collection()    # Main pipeline coordinator
├── rank_sections_by_relevance()     # Section ranking logic
├── calculate_relevance_score()      # Individual section scoring
├── create_refined_text()            # Contextual text generation
└── format_output()                  # Challenge-compliant JSON formatting
```

## Build and Run Instructions

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t round1b-solution:latest .
```

### Running the Solution
The container expects JSON files (outputs from Round 1A) in the input directory and takes persona and job as command line arguments:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none round1b-solution:latest python main.py "PhD Researcher in Computational Biology" "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
```

### Input Format
The system processes JSON files with the Round 1A format:
```json
{
    "title": "Document Title",
    "outline": [
        {"level": "H1", "text": "Introduction", "page": 1},
        {"level": "H2", "text": "Methodology", "page": 2}
    ]
}
```

### Output Format
Generates `challenge1b_output.json` with the exact format specified:
- **metadata**: Input documents, persona, job description, and timestamp
- **extracted_sections**: Top 10 ranked sections with importance ranking
- **subsection_analysis**: Top 5 sections with contextual refined text

## Performance Characteristics

- **Processing Time**: Optimized to complete within 60 seconds for typical JSON collections
- **Memory Usage**: Minimal memory footprint using efficient text processing
- **Model Size**: Uses lightweight spaCy model (~15MB) well under the 1GB constraint
- **CPU Only**: No GPU dependencies, efficient CPU-based processing

## Innovation Features

- **Context-Aware Refinement**: Generates meaningful refined text from section titles using document context
- **Adaptive Scoring**: Adjusts relevance calculation based on available NLP capabilities
- **Hierarchical Intelligence**: Understands document structure through heading levels
- **Robust Fallbacks**: Continues processing even with missing or corrupted JSON files

This solution transforms structured document outlines into intelligent, persona-driven insights while maintaining strict compliance with the challenge requirements and technical constraints.