import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple

from data_models import DocumentSection, ProcessingConfig
from pdf_extractor import PDFExtractor
from text_processor import TextProcessor
from relevance_scorer import RelevanceScorer

class PersonaDrivenDocumentIntelligence:
    """
    Enhanced system for Round 1B that extracts and ranks document sections
    based on persona and job-to-be-done requirements with improved accuracy.
    """
    
    def __init__(self):
        """Initialize the system with enhanced components"""
        self.config = ProcessingConfig()
        self.text_processor = TextProcessor()
        self.pdf_extractor = PDFExtractor(self.config)
        self.relevance_scorer = RelevanceScorer(self.text_processor)
        
        logging.info("PersonaDrivenDocumentIntelligence initialized")
        
    def process_document_collection(self, 
                                  pdf_paths: List[str], 
                                  persona: str, 
                                  job_to_be_done: str) -> Dict[str, Any]:
        """
        Main processing pipeline for Round 1B with enhanced accuracy
        """
        logging.info(f"Processing {len(pdf_paths)} documents for persona: {persona}")
        
        # Step 1: Extract sections from all documents
        all_sections = []
        document_stats = {}
        
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            logging.info(f"Extracting sections from: {doc_name}")
            
            sections = self.pdf_extractor.extract_sections_from_pdf(pdf_path)
            all_sections.extend(sections)
            
            document_stats[doc_name] = {
                'sections_extracted': len(sections),
                'total_content_length': sum(len(s.content) for s in sections)
            }
            
        logging.info(f"Extracted {len(all_sections)} total sections")
        
        # Log document statistics
        for doc_name, stats in document_stats.items():
            logging.info(f"{doc_name}: {stats['sections_extracted']} sections, "
                        f"{stats['total_content_length']} chars")
        
        # Step 2: Enhanced relevance ranking
        ranked_sections = self.relevance_scorer.rank_sections_by_relevance(
            all_sections, persona, job_to_be_done
        )
        
        # Step 3: Format output according to specification
        return self._format_output(
            pdf_paths, persona, job_to_be_done, ranked_sections
        )
    
    def _format_output(self, 
                      pdf_paths: List[str], 
                      persona: str, 
                      job_to_be_done: str, 
                      ranked_sections: List[Tuple[DocumentSection, float]]) -> Dict[str, Any]:
        """Format output according to challenge specification with enhanced content"""
        
        # Take top 1000 sections for extracted_sections
        top_sections = ranked_sections[:300]
        extracted_sections = []
        
        for i, (section, score) in enumerate(top_sections):
            extracted_sections.append({
                "document": section.document,
                "section_title": section.section_title,
                "importance_rank": i + 1,
                "page_number": section.page_number
            })
        
        # Take top 5 sections for detailed subsection analysis
        top_subsections = ranked_sections[:100]
        subsection_analysis = []
        
        for section, score in top_subsections:
            # Create enhanced refined text
            refined_text = self._create_enhanced_refined_text(section, persona, job_to_be_done)
            
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
                "processing_timestamp": datetime.now().isoformat(),
                "total_sections_analyzed": len(ranked_sections),
                "top_sections_selected": len(extracted_sections)
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    
    def _create_enhanced_refined_text(self, 
                                    section: DocumentSection, 
                                    persona: str, 
                                    job_to_be_done: str) -> str:
        """Create enhanced refined/summarized text from section content"""
        content = section.content.strip()
        title = section.section_title.strip()
        
        # If content is short enough, return with enhanced formatting
        if len(content) <= 250:
            return f"**{title}**: {content}"
        
        # For longer content, create an intelligent summary
        try:
            # Use enhanced summarization
            summary = self.text_processor.summarize_text(content, max_sentences=3)
            
            # Add context relevance
            relevant_keywords = self._extract_relevant_keywords(
                content, persona, job_to_be_done
            )
            
            # Enhance summary with key context if space allows
            if len(summary) < 300 and relevant_keywords:
                key_terms = ", ".join(relevant_keywords[:3])
                enhanced_summary = f"{summary} [Key topics: {key_terms}]"
                
                if len(enhanced_summary) <= 400:
                    summary = enhanced_summary
            
            # Ensure summary isn't too long
            if len(summary) > 400:
                summary = summary[:397] + "..."
                
            return f"**{title}**: {summary}"
            
        except Exception as e:
            logging.warning(f"Enhanced summarization failed: {e}")
            # Fallback: intelligent truncation
            truncated = self._intelligent_truncate(content, 300)
            return f"**{title}**: {truncated}"
    
    def _extract_relevant_keywords(self, 
                                 content: str, 
                                 persona: str, 
                                 job_to_be_done: str) -> List[str]:
        """Extract keywords most relevant to persona and job"""
        
        # Get domain keywords for content
        content_keywords = self.text_processor.extract_domain_keywords(content, persona, job_to_be_done)
        
        # Sort by relevance score and return top keywords
        sorted_keywords = sorted(content_keywords.items(), key=lambda x: x[1], reverse=True)
        
        return [keyword for keyword, score in sorted_keywords[:100] if score > 1.0]
    
    def _intelligent_truncate(self, text: str, max_length: int) -> str:
        """Intelligently truncate text at sentence boundaries"""
        if len(text) <= max_length:
            return text
            
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        
        # Find last sentence ending
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_length * 0.7:  # If we can keep at least 70% of text
            return truncated[:last_sentence_end + 1]
        else:
            # Truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."