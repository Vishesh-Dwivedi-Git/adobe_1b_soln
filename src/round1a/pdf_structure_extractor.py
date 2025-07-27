import fitz # PyMuPDF
import json
import time
from typing import Dict, List, Any
from .heading_detector import HeadingDetector
from .multilingual_handler import MultilingualHandler

class PDFStructureExtractor:
    """
    Extracts a structured outline (title and headings) from a PDF document.
    """
    def __init__(self):
        self.heading_detector = HeadingDetector()
        self.multilingual_handler = MultilingualHandler()

    def extract_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to extract the hierarchical structure from a PDF.
        """
        start_time = time.time()

        try:
            with fitz.open(pdf_path) as doc:
                title = self._extract_title(doc)
                outline = []
                for page_num, page in enumerate(doc):
                    # Get text with detailed formatting information
                    text_dict = page.get_text("dict")
                    page_text = page.get_text()

                    # Detect language for multilingual support
                    language = self.multilingual_handler.detect_language(page_text)

                    # Detect headings on the page
                    page_headings = self.heading_detector.detect_headings(
                        text_dict, page_num + 1, language
                    )
                    outline.extend(page_headings)

            processing_time = time.time() - start_time
            if processing_time > 10:
                print(f"Warning: Processing for {pdf_path} took {processing_time:.2f} seconds.")

            return {
                "title": title,
                "outline": outline
            }
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {"title": "Error Processing Document", "outline": []}


    def _extract_title(self, doc) -> str:
        """
        Extracts the document title, typically from the first page.
        Heuristic: Looks for the largest font size in the top 25% of the first page.
        """
        if not doc or len(doc) == 0:
            return "Untitled Document"

        first_page = doc[0]
        page_height = first_page.rect.height
        text_dict = first_page.get_text("dict")

        max_font_size = 0
        title_text = ""

        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    # Search in the top 25% of the page
                    if line["bbox"][1] < page_height * 0.25:
                        for span in line["spans"]:
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                title_text = span["text"].strip()

        return title_text or "Untitled Document"