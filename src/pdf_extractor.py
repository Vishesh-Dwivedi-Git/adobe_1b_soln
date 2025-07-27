import fitz  # PyMuPDF
import re
import numpy as np
import logging
from typing import List, Dict, Any
from collections import Counter

from data_models import DocumentSection, FontInfo, ProcessingConfig

class PDFExtractor:
    """Enhanced PDF text and structure extraction"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def extract_sections_from_pdf(self, pdf_path: str) -> List[DocumentSection]:
        """Extract meaningful sections from a PDF document with improved accuracy"""
        sections = []
        doc_name = pdf_path.split('/')[-1]  # Get filename only
        
        try:
            with fitz.open(pdf_path) as doc:
                # First pass: analyze document structure
                font_stats = self._analyze_document_fonts(doc)
                
                current_section = None
                
                for page_num, page in enumerate(doc):
                    text_dict = page.get_text("dict")
                    
                    # Process blocks in reading order (top to bottom)
                    blocks = sorted(text_dict.get("blocks", []), key=lambda b: b.get('bbox', [0, 0, 0, 0])[1])
                    
                    for block in blocks:
                        if block.get('type') != 0:  # Skip non-text blocks
                            continue
                            
                        # Process lines in reading order
                        lines = sorted(block.get("lines", []), key=lambda l: l.get('bbox', [0, 0, 0, 0])[1])
                        
                        for line in lines:
                            line_text = self._extract_line_text(line)
                            if not line_text.strip():
                                continue
                                
                            # Get enhanced line formatting information
                            font_info = self._get_enhanced_line_formatting(line)
                            position = line.get('bbox', [0, 0, 0, 0])[1]  # Y coordinate
                            
                            # Enhanced heading detection
                            is_heading = self._is_section_heading_enhanced(
                                line_text, font_info, font_stats, page_num
                            )
                            
                            if is_heading:
                                # Save previous section if it exists and is valid
                                if current_section and self._is_valid_section(current_section):
                                    sections.append(current_section)
                                
                                # Start new section
                                current_section = DocumentSection(
                                    document=doc_name,
                                    section_title=self._clean_section_title(line_text),
                                    content="",
                                    page_number=page_num + 1,
                                    font_size=font_info.size,
                                    is_bold=font_info.bold,
                                    position=position
                                )
                            else:
                                # Add content to current section
                                if current_section:
                                    # Clean and append text
                                    clean_text = self._clean_content_text(line_text)
                                    if clean_text:
                                        current_section.content += clean_text + " "
                                else:
                                    # Create a default section for content without heading
                                    clean_text = self._clean_content_text(line_text)
                                    if clean_text:
                                        current_section = DocumentSection(
                                            document=doc_name,
                                            section_title="Introduction",
                                            content=clean_text + " ",
                                            page_number=page_num + 1,
                                            font_size=font_info.size,
                                            is_bold=font_info.bold,
                                            position=position
                                        )
                
                # Add the last section
                if current_section and self._is_valid_section(current_section):
                    sections.append(current_section)
                    
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            
        return self._post_process_sections(sections)
    
    def _analyze_document_fonts(self, doc) -> Dict[str, Any]:
        """Analyze font usage patterns in the document"""
        font_sizes = []
        font_names = []
        bold_flags = []
        
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get('type') != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span.get("size", 12))
                        font_names.append(span.get("font", ""))
                        bold_flags.append(bool(span.get("flags", 0) & (1 << 4)))
        
        if not font_sizes:
            return {"body_size": 12, "heading_threshold": 14, "common_fonts": []}
            
        # Calculate statistics
        font_counter = Counter(font_sizes)
        body_size = font_counter.most_common(1)[0][0]  # Most common font size
        heading_threshold = body_size + 1.5
        
        return {
            "body_size": body_size,
            "heading_threshold": heading_threshold,
            "common_fonts": Counter(font_names).most_common(5),
            "avg_size": np.mean(font_sizes),
            "max_size": max(font_sizes),
            "bold_ratio": sum(bold_flags) / len(bold_flags) if bold_flags else 0
        }
    
    def _extract_line_text(self, line: Dict) -> str:
        """Extract text from a line object"""
        text_parts = []
        for span in line.get("spans", []):
            text_parts.append(span.get("text", ""))
        return "".join(text_parts)
    
    def _get_enhanced_line_formatting(self, line: Dict) -> FontInfo:
        """Get enhanced formatting information for a line"""
        sizes = []
        fonts = []
        flags = []
        colors = []
        
        for span in line.get("spans", []):
            sizes.append(span.get("size", 12))
            fonts.append(span.get("font", ""))
            flags.append(span.get("flags", 0))
            colors.append(span.get("color", 0))
        
        avg_size = np.mean(sizes) if sizes else 12
        is_bold = any(flag & (1 << 4) for flag in flags) if flags else False
        is_italic = any(flag & (1 << 1) for flag in flags) if flags else False
        
        return FontInfo(
            size=avg_size,
            bold=is_bold,
            fonts=fonts,
            italic=is_italic,
            color=str(colors[0]) if colors else ""
        )
    
    def _is_section_heading_enhanced(self, text: str, font_info: FontInfo, 
                                   font_stats: Dict, page_num: int) -> bool:
        """Enhanced heading detection with multiple signals"""
        text = text.strip()
        
        # Basic filters
        if len(text) < self.config.min_heading_words or len(text) > 200:
            return False
            
        if self._is_artifact_text(text):
            return False
        
        word_count = len(text.split())
        if word_count > self.config.max_heading_words:
            return False
            
        # Scoring system
        score = 0
        
        # Font size scoring (most important)
        size_diff = font_info.size - font_stats["body_size"]
        if size_diff >= 3:
            score += 4
        elif size_diff >= 2:
            score += 3
        elif size_diff >= 1:
            score += 2
        elif size_diff >= 0.5:
            score += 1
            
        # Bold formatting
        if font_info.bold:
            score += 2
            
        # Structural patterns
        if re.match(r'^\s*\d+(\.\d+)*[\.\s]', text):  # Numbered sections
            score += 3
        elif re.match(r'^\s*[A-Z][A-Z\s]{2,}$', text):  # ALL CAPS
            score += 2
        elif text.istitle():  # Title Case
            score += 1
            
        # Content-based signals
        heading_keywords = {
            'abstract', 'introduction', 'background', 'methodology', 'method',
            'results', 'discussion', 'conclusion', 'conclusions', 'references', 
            'summary', 'overview', 'analysis', 'findings', 'recommendations', 
            'appendix', 'literature', 'review', 'approach', 'evaluation',
            'implementation', 'design', 'architecture', 'framework'
        }
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in heading_keywords):
            score += 2
            
        # Position-based scoring (first page headings are more likely)
        if page_num == 0:
            score += 1
            
        # Length-based scoring (shorter text more likely to be heading)
        if word_count <= 5:
            score += 1
        elif word_count <= 8:
            score += 0.5
            
        # Punctuation patterns
        if text.endswith(':'):
            score += 1
        elif text.count('.') > word_count * 0.3:  # Too many periods
            score -= 2
            
        return score >= 4
    
    def _is_artifact_text(self, text: str) -> bool:
        """Enhanced artifact detection"""
        patterns = [
            r'^\s*page\s+\d+\s*$',
            r'^\s*\d+\s*$',
            r'^\s*figure\s+\d+',
            r'^\s*table\s+\d+',
            r'^\s*fig\.\s*\d+',
            r'^\s*tab\.\s*\d+',
            r'copyright|©|\(c\)',
            r'^\s*[.\-_–—]+\s*$',
            r'^\s*www\.',
            r'^\s*http[s]?://',
            r'^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            r'^\s*[A-Z]{2,}\s*\d+\s*$',  # Code patterns
        ]
        
        return any(re.match(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _clean_section_title(self, text: str) -> str:
        """Clean section title text"""
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove trailing colons or periods
        text = re.sub(r'[:\.]$', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _clean_content_text(self, text: str) -> str:
        """Clean content text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Skip if too short or looks like artifact
        if len(text) < 3 or self._is_artifact_text(text):
            return ""
            
        return text
    
    def _is_valid_section(self, section: DocumentSection) -> bool:
        """Enhanced section validation"""
        content = section.content.strip()
        
        # Length checks
        if len(content) < self.config.min_section_length:
            return False
        if len(content) > self.config.max_section_length:
            return False
            
        # Content quality checks
        if self._is_artifact_text(section.section_title):
            return False
            
        # Check if content is mostly meaningful text
        words = content.split()
        if len(words) < 5:
            return False
            
        # Check for excessive repetition
        word_freq = Counter(words)
        if word_freq.most_common(1)[0][1] > len(words) * 0.3:
            return False
            
        return True
    
    def _post_process_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Post-process sections to improve quality"""
        if not sections:
            return sections
            
        # Remove duplicate sections
        seen_titles = set()
        unique_sections = []
        
        for section in sections:
            title_key = section.section_title.lower().strip()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sections.append(section)
        
        # Sort by document order (page, then position)
        unique_sections.sort(key=lambda s: (s.page_number, s.position))
        
        return unique_sections