# Adobe Hackathon: Connecting the Dots - Solution (Enhanced)

This project is an enhanced solution for the Adobe India Hackathon "Connecting the Dots" challenge. It provides an end-to-end offline PDF document intelligence system with improved NLP capabilities.

## Demo
<video controls src="2025-07-27 21-52-15.mkv" title="Title"></video>
## Features

- **Round 1A: Document Structure Extraction**
  - Extracts title and hierarchical headings (H1, H2, H3) from PDFs using a combination of font analysis and linguistic features from `spaCy`.
  - Outputs a structured JSON file.
  - Includes bonus multilingual support.
  - Optimized for speed (under 10 seconds for a 50-page document).

- **Round 1B: Persona-Driven Document Intelligence**
  - Analyzes a collection of related PDFs.
  - Ranks document sections by **semantic relevance** and importance based on a user persona and a specific task, using `spaCy` word embeddings.
  - Outputs a ranked list of relevant sections in JSON format.
  - Operates completely offline.

## Technology Stack

- **PDF Processing:** PyMuPDF (`fitz`)
- **Machine Learning & NLP:** scikit-learn (for Decision Trees), `spaCy` (for embeddings and linguistic features)
- **Language Detection:** `langdetect`
- **Containerization:** Docker

## Usage Instructions

### Building and Running with Docker (Recommended)

1.  **Build the Docker image:**
    This will also download the necessary `spaCy` model.
    ```bash
    docker build -t adobe-hackathon-solution-enhanced . 
    ```

2.  **Prepare Input Files:**
    Place your PDF files inside the `input/` directory.

3.  **Run Round 1A (Structure Extraction):**
    This command processes `document.pdf` from the `input` folder and saves the result in the `output` folder.

    ```bash
    docker run --rm \
      -v "$(pwd)/input:/app/input" \
      -v "$(pwd)/output:/app/output" \
      adobe-hackathon-solution-enhanced \
      python main.py 1a /app/input/document.pdf
    ```

4.  **Run Round 1B (Document Intelligence):**
    This command processes all PDFs in the `input` directory for the "Data Scientist" persona analyzing "methodology".

    ```bash
    docker run --rm \
      -v "$(pwd)/input:/app/input" \
      -v "$(pwd)/output:/app/output" \
      adobe-hackathon-solution-enhanced \
      python main.py 1b /app/input "Data Scientist" "methodology analysis"
    ```

### Local Development

1.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

2.  **Prepare Input Files:**
    Place your PDF files inside the `input/` directory.

3.  **Run Round 1A:**
    ```bash
    python src/main.py 1a input/your_document.pdf
    ```

4.  **Run Round 1B:**
    ```bash
    python src/main.py 1b input/ "Data Scientist" "methodology analysis"
    ```

## How It Works (Enhanced)

### Round 1A: Structure Extraction

The system processes PDFs page by page. For each page, it extracts text along with font information. A Decision Tree classifier, now trained with **linguistic features** (like part-of-speech counts) from `spaCy`, identifies potential headings. This reduces reliance on font size alone and improves accuracy.

### Round 1B: Document Intelligence

Instead of TF-IDF, the enhanced system now uses **`spaCy`'s word embeddings** to represent the semantic meaning of the query and document sections. It calculates the **cosine similarity** between these embedding vectors to determine a more accurate `relevance_score`. This allows the system to understand context and meaning, not just keyword matching.
