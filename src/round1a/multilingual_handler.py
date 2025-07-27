from langdetect import detect
from typing import List

class MultilingualHandler:
    """
    Handles language detection for multilingual documents.
    This is a bonus requirement for Round 1A.
    """
    def __init__(self):
        self.supported_languages: List[str] = [
            'en', 'es', 'fr', 'de', 'zh-cn', 'ja', 'ko', 'hi', 'ar'
        ]

    def detect_language(self, text: str) -> str:
        """
        Detects the language of a given text snippet.
        Defaults to English if detection fails or text is too short.
        """
        try:
            # If text is very short, detection can be unreliable
            if len(text.strip()) < 20:
                return 'en'
            return detect(text)
        except Exception:
            # Default to English in case of any errors
            return 'en'

    def process_multilingual_text(self, text: str, language: str) -> str:
        """
        Processes text based on its detected language.
        (Placeholder for language-specific text processing)
        """
        # CJK (Chinese, Japanese, Korean) languages often don't use spaces
        if language in ['zh-cn', 'ja', 'ko']:
            return text.strip()
        # Handle Right-to-Left languages
        elif language in ['ar', 'he']:
            return text.strip()
        # Default for LTR languages
        else:
            return text.strip()

