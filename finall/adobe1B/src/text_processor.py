import spacy
import re
import logging
from typing import List, Dict, Set, Optional
from collections import Counter
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using basic similarity methods")

class TextProcessor:
    """Enhanced text processing and similarity computation"""
    
    def __init__(self):
        self.nlp = None
        self.has_vectors = False
        self.tfidf = None
        self.stop_words = self._get_stop_words()
        self._setup_nlp()
        if SKLEARN_AVAILABLE:
            self._setup_tfidf()
        
    def _get_stop_words(self) -> Set[str]:
        """Get comprehensive stop words list"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours',
            'hers', 'ours', 'theirs', 'also', 'such', 'very', 'more', 'most',
            'some', 'any', 'many', 'much', 'few', 'little', 'other', 'another',
            'each', 'every', 'all', 'both', 'either', 'neither', 'not', 'no',
            'nor', 'so', 'than', 'too', 'only', 'just', 'now', 'then', 'here',
            'there', 'where', 'when', 'why', 'how', 'what', 'which', 'who', 'whom'
        }
        
    def _setup_nlp(self):
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
                self.nlp = None
                self.has_vectors = False
                logging.warning("No spaCy model available, using basic text processing")
    
    def _setup_tfidf(self):
        """Setup TF-IDF vectorizer as fallback for similarity"""
        if SKLEARN_AVAILABLE:
            self.tfidf = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        # Try spaCy similarity first
        if self.nlp and self.has_vectors:
            try:
                # Limit text length for performance
                doc1 = self.nlp(text1[:1000])
                doc2 = self.nlp(text2[:1000])
                similarity = doc1.similarity(doc2)
                return max(0.0, min(1.0, similarity))  # Clamp to [0,1]
            except Exception as e:
                logging.warning(f"SpaCy similarity failed: {e}")
        
        # Fallback to TF-IDF similarity
        return self._tfidf_similarity(text1, text2)
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based similarity"""
        try:
            # Fit and transform both texts
            corpus = [text1[:1000], text2[:1000]]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            logging.warning(f"TF-IDF similarity failed: {e}")
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fallback Jaccard similarity for keywords"""
        words1 = set(self.extract_key_terms(text1.lower()))
        words2 = set(self.extract_key_terms(text2.lower()))
        
        if not words1 and not words2:
            return 0.0
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text with improved filtering"""
        if not text.strip():
            return []
            
        # Enhanced stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours',
            'his', 'hers', 'ours', 'theirs', 'also', 'such', 'very', 'more', 'most',
            'some', 'any', 'many', 'much', 'few', 'little', 'other', 'another'
        }
        
        # Use spaCy if available for better token processing
        if self.nlp:
            try:
                doc = self.nlp(text.lower())
                terms = []
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        not token.is_space and
                        len(token.text) > 2 and
                        token.text not in stop_words and
                        token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                        terms.append(token.lemma_)
                return terms
            except:
                pass
        
        # Fallback to regex-based extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms
    
    def extract_domain_keywords(self, text: str, persona: str, job: str) -> Dict[str, float]:
        """Extract domain-specific keywords with weights"""
        all_text = f"{text} {persona} {job}".lower()
        terms = self.extract_key_terms(all_text)
        
        # Count term frequency
        term_freq = Counter(terms)
        
        # Domain-specific keyword categories with weights
        domain_categories = {
            'technical': {
                'keywords': ['algorithm', 'method', 'approach', 'technique', 'system', 
                           'model', 'framework', 'implementation', 'analysis', 'evaluation',
                           'performance', 'optimization', 'architecture', 'design'],
                'weight': 1.5
            },
            'academic': {
                'keywords': ['research', 'study', 'literature', 'review', 'methodology',
                           'results', 'findings', 'conclusion', 'hypothesis', 'experiment',
                           'data', 'analysis', 'survey', 'investigation'],
                'weight': 1.3
            },
            'business': {
                'keywords': ['strategy', 'management', 'process', 'business', 'market',
                           'revenue', 'profit', 'investment', 'growth', 'competitive',
                           'analysis', 'report', 'financial', 'budget'],
                'weight': 1.2
            }
        }
        
        # Calculate weighted scores
        keyword_scores = {}
        for term, freq in term_freq.items():
            score = freq
            
            # Apply domain weights
            for category, info in domain_categories.items():
                if term in info['keywords']:
                    score *= info['weight']
                    break
            
            keyword_scores[term] = score
        
        return keyword_scores
    
    def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Create a summary of the text"""
        if not text.strip():
            return ""
            
        # Use spaCy for sentence segmentation if available
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except:
                sentences = self._simple_sentence_split(text)
        else:
            sentences = self._simple_sentence_split(text)
        
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        
        # Simple extractive summarization - take first and most informative sentences
        if len(sentences) > max_sentences:
            # Always include first sentence
            summary_sentences = [sentences[0]]
            
            # Score remaining sentences by keyword density
            scored_sentences = []
            key_terms = set(self.extract_key_terms(text.lower()))
            
            for i, sentence in enumerate(sentences[1:], 1):
                sentence_terms = set(self.extract_key_terms(sentence.lower()))
                score = len(sentence_terms.intersection(key_terms)) / len(sentence_terms) if sentence_terms else 0
                scored_sentences.append((score, i, sentence))
            
            # Sort by score and take top sentences
            scored_sentences.sort(reverse=True)
            for _, _, sentence in scored_sentences[:max_sentences-1]:
                summary_sentences.append(sentence)
        
        return " ".join(summary_sentences)
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting fallback"""
        # Split on sentence endings, but be careful with abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]