#!/usr/bin/env python3

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import re
import spacy
import numpy as np
from collections import Counter
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)')

@dataclass
class DocumentSection:
    """Data class to hold document section information"""
    document: str
    section_title: str
    content: str
    page_number: int
    font_size: float
    is_bold: bool
    position: float  # Y-coordinate for ordering

class PersonaDrivenDocumentIntelligence:
    """
    Main system for Round 1B that extracts and ranks document sections
    based on persona and job-to-be-done requirements.
    """
    
    def __init__(self):
        """Initialize the system with required components"""
        self.min_section_length = 30
        self.max_section_length = 2000
        self.setup_nlp()
        
    def setup_nlp(self):
        """Setup spaCy NLP with fallback options"""
        try:
            self.nlp = spacy.load("en_core_web_md")
            self.has_vectors = True
            logging.info("Loaded en_core_web_md model")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.has_vectors = False
                logging.info("Loaded en_core_web_sm model (no vectors)")
            except OSError:
                # Fallback to basic text processing
                self.nlp = None
                self.has_vectors = False
                logging.warning("No spaCy model available, using basic text processing")

    def process_document_collection(self, 
                                  pdf_paths: List[str], 
                                  persona: str, 
                                  job_to_be_done: str) -> Dict[str, Any]:
        """
        Main processing pipeline for Round 1B
        """
        logging.info(f"Processing {len(pdf_paths)} documents for persona: {persona}")
        
        # Step 1: Extract sections from all documents
        all_sections = []
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            logging.info(f"Extracting sections from: {doc_name}")
            sections = self.extract_sections_from_pdf(pdf_path)
            all_sections.extend(sections)
            
        logging.info(f"Extracted {len(all_sections)} total sections")
        
        # Step 2: Rank sections based on relevance to persona and job
        ranked_sections = self.rank_sections_by_relevance(
            all_sections, persona, job_to_be_done
        )
        
        # Step 3: Format output according to challenge specification
        return self.format_output(
            pdf_paths, persona, job_to_be_done, ranked_sections
        )
    
    def extract_sections_from_pdf(self, pdf_path: str) -> List[DocumentSection]:
        """Extract meaningful sections from a PDF document"""
        sections = []
        doc_name = os.path.basename(pdf_path)
        
        try:
            with fitz.open(pdf_path) as doc:
                current_section = None
                
                for page_num, page in enumerate(doc):
                    text_dict = page.get_text("dict")
                    
                    for block in text_dict.get("blocks", []):
                        if block.get('type') != 0:  # Skip non-text blocks
                            continue
                            
                        for line in block.get("lines", []):
                            line_text = self.extract_line_text(line)
                            if not line_text.strip():
                                continue
                                
                            # Get line formatting information
                            font_info = self.get_line_formatting(line)
                            position = line.get('bbox', [0, 0, 0, 0])[1]  # Y coordinate
                            
                            # Determine if this is a section heading
                            is_heading = self.is_section_heading(
                                line_text, font_info['size'], font_info['bold']
                            )
                            
                            if is_heading:
                                # Save previous section if it exists and is valid
                                if current_section and self.is_valid_section(current_section):
                                    sections.append(current_section)
                                
                                # Start new section
                                current_section = DocumentSection(
                                    document=doc_name,
                                    section_title=line_text.strip(),
                                    content="",
                                    page_number=page_num + 1,
                                    font_size=font_info['size'],
                                    is_bold=font_info['bold'],
                                    position=position
                                )
                            else:
                                # Add content to current section
                                if current_section:
                                    current_section.content += line_text + " "
                                else:
                                    # Create a default section for content without heading
                                    current_section = DocumentSection(
                                        document=doc_name,
                                        section_title="Introduction",
                                        content=line_text + " ",
                                        page_number=page_num + 1,
                                        font_size=font_info['size'],
                                        is_bold=font_info['bold'],
                                        position=position
                                    )
                
                # Add the last section
                if current_section and self.is_valid_section(current_section):
                    sections.append(current_section)
                    
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            
        return sections
    
    def extract_line_text(self, line: Dict) -> str:
        """Extract text from a line object"""
        text_parts = []
        for span in line.get("spans", []):
            text_parts.append(span.get("text", ""))
        return "".join(text_parts)
    
    def get_line_formatting(self, line: Dict) -> Dict[str, Any]:
        """Get formatting information for a line"""
        sizes = []
        fonts = []
        flags = []
        
        for span in line.get("spans", []):
            sizes.append(span.get("size", 12))
            fonts.append(span.get("font", ""))
            flags.append(span.get("flags", 0))
        
        avg_size = np.mean(sizes) if sizes else 12
        is_bold = any(flag & (1 << 4) for flag in flags) if flags else False
        
        return {
            'size': avg_size,
            'bold': is_bold,
            'fonts': fonts
        }
    
    def is_section_heading(self, text: str, font_size: float, is_bold: bool) -> bool:
        """Determine if a line of text is likely a section heading"""
        text = text.strip()
        
        # Skip very short or very long text
        if len(text) < 3 or len(text) > 150:
            return False
            
        # Skip obvious non-headings
        if self.is_artifact_text(text):
            return False
            
        # Heading indicators
        word_count = len(text.split())
        
        # Style-based detection
        style_score = 0
        if font_size > 14:
            style_score += 3
        elif font_size > 12:
            style_score += 2
            
        if is_bold:
            style_score += 2
            
        # Content-based detection
        content_score = 0
        
        # Numbered headings
        if re.match(r'^\s*\d+(\.\d+)*[\.\s]', text):
            content_score += 3
            
        # Title case or all caps
        if text.istitle() or text.isupper():
            content_score += 1
            
        # Common heading keywords
        heading_keywords = {
            'abstract', 'introduction', 'background', 'methodology', 'method',
            'results', 'discussion', 'conclusion', 'references', 'summary',
            'overview', 'analysis', 'findings', 'recommendations', 'appendix'
        }
        
        if any(keyword in text.lower() for keyword in heading_keywords):
            content_score += 2
            
        # Short text is more likely to be heading
        if word_count <= 8:
            content_score += 1
            
        return (style_score + content_score) >= 4
    
    def is_artifact_text(self, text: str) -> bool:
        """Check if text is likely an artifact (page numbers, etc.)"""
        patterns = [
            r'^\s*page\s+\d+\s*$',
            r'^\s*\d+\s*$',
            r'^\s*figure\s+\d+',
            r'^\s*table\s+\d+',
            r'copyright|©',
            r'^\s*[.\-_–—]+\s*$'
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def is_valid_section(self, section: DocumentSection) -> bool:
        """Check if a section is valid for inclusion"""
        content = section.content.strip()
        return (
            len(content) >= self.min_section_length and
            len(content) <= self.max_section_length and
            not self.is_artifact_text(section.section_title)
        )
    
    def rank_sections_by_relevance(self, 
                                 sections: List[DocumentSection], 
                                 persona: str, 
                                 job_to_be_done: str) -> List[Tuple[DocumentSection, float]]:
        """Rank sections by relevance to persona and job requirements"""
        if not sections:
            return []
            
        # Create query from persona and job
        query = f"{persona} {job_to_be_done}"
        
        scored_sections = []
        
        for section in sections:
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(
                section, query, persona, job_to_be_done
            )
            scored_sections.append((section, relevance_score))
        
        # Sort by relevance score (descending)
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        return scored_sections
    
    def calculate_relevance_score(self, 
                                section: DocumentSection, 
                                query: str, 
                                persona: str, 
                                job_to_be_done: str) -> float:
        """Calculate relevance score for a section"""
        score = 0.0
        
        # Combine section title and content for analysis
        full_text = f"{section.section_title} {section.content}".lower()
        query_lower = query.lower()
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Semantic similarity using spaCy (if available)
        if self.nlp and self.has_vectors:
            try:
                text_doc = self.nlp(full_text[:1000])  # Limit text for performance
                query_doc = self.nlp(query_lower)
                semantic_score = text_doc.similarity(query_doc)
                score += semantic_score * 0.6
            except:
                pass  # Fall back to keyword matching
        
        # Keyword matching scores
        keyword_score = 0.0
        
        # Extract key terms from persona and job
        persona_terms = self.extract_key_terms(persona_lower)
        job_terms = self.extract_key_terms(job_lower)
        
        # Score based on persona term matches
        for term in persona_terms:
            if term in full_text:
                keyword_score += 0.1
                
        # Score based on job term matches  
        for term in job_terms:
            if term in full_text:
                keyword_score += 0.15
                
        score += min(keyword_score, 0.4)  # Cap keyword contribution
        
        # Title relevance bonus
        title_lower = section.section_title.lower()
        if any(term in title_lower for term in persona_terms + job_terms):
            score += 0.1
            
        # Section length normalization (prefer substantial sections)
        content_length = len(section.content)
        if 100 <= content_length <= 1000:
            score += 0.05
        elif content_length > 1000:
            score += 0.03
            
        return min(score, 1.0)  # Cap at 1.0
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return key_terms
    
    def format_output(self, 
                     pdf_paths: List[str], 
                     persona: str, 
                     job_to_be_done: str, 
                     ranked_sections: List[Tuple[DocumentSection, float]]) -> Dict[str, Any]:
        """Format output according to challenge specification"""
        
        # Take top all sections for extracted_sections
        top_sections = ranked_sections[:]
        extracted_sections = []
        
        for i, (section, score) in enumerate(top_sections):
            extracted_sections.append({
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": i + 1,
                "page_number": section.page_number
            })
        
        # Take top 5 sections for detailed subsection analysis
        top_subsections = ranked_sections[:5]
        subsection_analysis = []
        
        for section, score in top_subsections:
            # Create refined text (summarized content)
            refined_text = self.create_refined_text(section)
            
            subsection_analysis.append({
                "document": section.document,
                "refined_text": refined_text,
                "page_number": section.page_number
            })
        
        return {
            "metadata": {
                "input_documents": [os.path.basename(path) for path in pdf_paths],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    
    def create_refined_text(self, section: DocumentSection) -> str:
        """Create refined/summarized text from section content"""
        content = section.content.strip()
        
        # If content is short enough, return as-is with title
        if len(content) <= 300:
            return f"{section.section_title}: {content}"
        
        # For longer content, create a summary
        if self.nlp:
            try:
                doc = self.nlp(content)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                # Take first 2-3 sentences for summary
                summary_sentences = sentences[:3]
                summary = " ".join(summary_sentences)
                
                # Ensure summary isn't too long
                if len(summary) > 400:
                    summary = summary[:400] + "..."
                    
                return f"{section.section_title}: {summary}"
            except:
                pass
        
        # Fallback: just truncate content
        truncated = content[:300] + "..." if len(content) > 300 else content
        return f"{section.section_title}: {truncated}"


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

