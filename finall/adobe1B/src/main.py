#!/usr/bin/env python3

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List

from document_processor import PersonaDrivenDocumentIntelligence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main execution function for Round 1B"""
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files in input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logging.error("No PDF files found in input directory")
        return
    
    pdf_paths = [str(pdf_path) for pdf_path in pdf_files]
    
    # Get persona and job from environment variables or use defaults
    persona = os.environ.get('PERSONA', 'Food Contractor')
    job_to_be_done = os.environ.get('JOB_TO_BE_DONE', 
                                   'Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items.')
    
    logging.info(f"Processing {len(pdf_paths)} PDF files")
    logging.info(f"Persona: {persona}")
    logging.info(f"Job to be done: {job_to_be_done}")
    
    # Initialize and run the system
    logging.info("Starting Round 1B Document Intelligence System")
    start_time = time.time()
    
    system = PersonaDrivenDocumentIntelligence()
    result = system.process_document_collection(pdf_paths, persona, job_to_be_done)
    
    processing_time = time.time() - start_time
    logging.info(f"Processing completed in {processing_time:.2f} seconds")
    
    # Save result
    output_path = output_dir / "challenge1b_output.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")

if __name__ == "__main__":
    main()