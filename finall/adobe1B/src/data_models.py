from dataclasses import dataclass
from typing import List, Dict, Any

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
    word_count: int = 0
    
    def __post_init__(self):
        """Calculate word count after initialization"""
        self.word_count = len(self.content.split())

@dataclass
class FontInfo:
    """Data class for font information"""
    size: float
    bold: bool
    fonts: List[str]
    italic: bool = False
    color: str = ""

@dataclass 
class ProcessingConfig:
    """Configuration for document processing"""
    min_section_length: int = 30
    max_section_length: int = 2000
    heading_min_font_size: float = 12.0
    content_max_font_size: float = 14.0
    max_heading_words: int = 15
    min_heading_words: int = 1