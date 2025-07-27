import re
from typing import List, Dict, Any
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import spacy

class HeadingDetector:
    """
    Detects headings in a PDF page using a combination of a decision tree
    classifier, font style heuristics, and spaCy linguistic features.
    """
    def __init__(self):
        self.classifier = self._build_classifier()
        # Load the small English model for spaCy
        self.nlp = spacy.load("en_core_web_sm")

    def _build_classifier(self) -> DecisionTreeClassifier:
        """
        Builds and trains a simple Decision Tree classifier for heading detection.
        The features are [font_size, is_bold, is_uppercase, text_length, num_nouns, num_verbs].
        """
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=10,
            random_state=42
        )

        # Features: [font_size, is_bold, is_uppercase, text_length, num_nouns, num_verbs]
        X_train = np.array([
            [16, 1, 1, 15, 2, 0],  # H1 heading
            [14, 1, 0, 25, 3, 1],  # H2 heading
            [12, 0, 0, 100, 10, 5], # Body text
            [18, 1, 1, 20, 3, 0],  # Title/H1
            [12, 0, 0, 150, 15, 8], # Body text
            [12, 1, 0, 30, 4, 1],  # H3 heading
        ])
        # Labels: 1 for heading, 0 for body text
        y_train = np.array([1, 1, 0, 1, 0, 1])

        clf.fit(X_train, y_train)
        return clf

    def detect_headings(self, text_dict: Dict, page_num: int, language: str = 'en') -> List[Dict]:
        """
        Detects headings from the dictionary representation of a page's text.
        """
        headings = []
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    font_size = 0
                    is_bold = False
                    for span in line["spans"]:
                        line_text += span["text"]
                        font_size = max(font_size, span["size"])
                        is_bold = is_bold or ("bold" in span["font"].lower())

                    line_text = line_text.strip()
                    if 3 <= len(line_text) <= 150:
                        features = self._extract_features(line_text, font_size, is_bold)
                        if self._is_heading(features):
                            level = self._determine_heading_level(font_size, is_bold, line_text)
                            headings.append({
                                "heading": line_text,
                                "level": level,
                                "page": page_num
                            })
        return headings

    def _extract_features(self, text: str, font_size: float, is_bold: bool) -> List[float]:
        """
        Extracts features from a line of text for the classifier, now including linguistic features.
        """
        doc = self.nlp(text)
        num_nouns = len([token for token in doc if token.pos_ == "NOUN"])
        num_verbs = len([token for token in doc if token.pos_ == "VERB"])
        return [
            font_size,
            1 if is_bold else 0,
            1 if text.isupper() else 0,
            len(text),
            num_nouns,
            num_verbs
        ]

    def _is_heading(self, features: List[float]) -> bool:
        """
        Uses the trained classifier to predict if a line is a heading.
        """
        prediction = self.classifier.predict([features])
        return prediction[0] == 1

    def _determine_heading_level(self, font_size: float, is_bold: bool, text: str) -> int:
        """
        Determines the heading level (H1, H2, H3) based on font size, style, and text properties.
        """
        # More weight to larger font sizes for H1
        if font_size >= 18 or (font_size >= 16 and text.isupper()):
            return 1  # H1
        elif font_size >= 14:
            return 2  # H2
        elif font_size >= 12 and is_bold:
            return 3  # H3
        else:
            return 3 # Default to H3 for other detected headings
