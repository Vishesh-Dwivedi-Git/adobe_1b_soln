import re
import logging
from typing import List, Tuple, Dict
from collections import Counter

from data_models import DocumentSection
from text_processor import TextProcessor

class RelevanceScorer:
    """Enhanced relevance scoring for document sections"""
    
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        
    def rank_sections_by_relevance(self, 
                                 sections: List[DocumentSection], 
                                 persona: str, 
                                 job_to_be_done: str) -> List[Tuple[DocumentSection, float]]:
        """Rank sections by relevance to persona and job requirements"""
        if not sections:
            return []
            
        logging.info(f"Ranking {len(sections)} sections for relevance")
        
        # Pre-process persona and job descriptions
        persona_processed = self._preprocess_query_text(persona)
        job_processed = self._preprocess_query_text(job_to_be_done)
        combined_query = f"{persona_processed} {job_processed}"
        
        # Extract key domain terms
        persona_keywords = self.text_processor.extract_domain_keywords(persona, "", "")
        job_keywords = self.text_processor.extract_domain_keywords(job_to_be_done, "", "")
        
        scored_sections = []
        
        for section in sections:
            relevance_score = self._calculate_enhanced_relevance_score(
                section, combined_query, persona_processed, job_processed,
                persona_keywords, job_keywords
            )
            scored_sections.append((section, relevance_score))
        
        # Sort by relevance score (descending)
        scored_sections.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"Top 5 section scores: {[(s.section_title[:30], round(score, 3)) for s, score in scored_sections]}")
        
        return scored_sections
    
    def _preprocess_query_text(self, text: str) -> str:
        """Preprocess query text for better matching"""
        # Convert to lowercase and clean
        text = text.lower().strip()
        
        # Expand common abbreviations
        abbreviations = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'api': 'application programming interface',
            'ui': 'user interface',
            'ux': 'user experience'
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        
        return text
    
    def _calculate_enhanced_relevance_score(self, 
                                          section: DocumentSection, 
                                          combined_query: str, 
                                          persona: str, 
                                          job: str,
                                          persona_keywords: Dict[str, float],
                                          job_keywords: Dict[str, float]) -> float:
        """Calculate enhanced relevance score using multiple signals"""
        
        # Combine section title and content for analysis
        section_text = f"{section.section_title} {section.content}".lower()
        title_text = section.section_title.lower()
        content_text = section.content.lower()
        
        total_score = 0.0
        
        # 1. Semantic Similarity (40% weight)
        semantic_score = self.text_processor.calculate_semantic_similarity(
            section_text[:1500], combined_query  # Limit for performance
        )
        total_score += semantic_score * 0.4
        
        # 2. Keyword Matching (30% weight)
        keyword_score = self._calculate_keyword_matching_score(
            section_text, persona_keywords, job_keywords
        )
        total_score += keyword_score * 0.3
        
        # 3. Title Relevance (15% weight)
        title_score = self._calculate_title_relevance(
            title_text, persona, job, persona_keywords, job_keywords
        )
        total_score += title_score * 0.15
        
        # 4. Content Quality and Structure (10% weight)
        quality_score = self._calculate_content_quality_score(section)
        total_score += quality_score * 0.1
        
        # 5. Domain-specific Bonuses (5% weight)
        domain_score = self._calculate_domain_specific_score(
            section_text, persona, job
        )
        total_score += domain_score * 0.05
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def _calculate_keyword_matching_score(self, 
                                        section_text: str,
                                        persona_keywords: Dict[str, float],
                                        job_keywords: Dict[str, float]) -> float:
        """Calculate keyword matching score with TF-IDF weighting"""
        
        # Extract section keywords
        section_keywords = self.text_processor.extract_domain_keywords(section_text, "", "")
        
        persona_score = 0.0
        job_score = 0.0
        
        # Calculate weighted matches for persona keywords
        for keyword, weight in persona_keywords.items():
            if keyword in section_keywords:
                # Boost score based on frequency in section
                section_freq = section_keywords[keyword]
                persona_score += min(weight * section_freq * 0.1, 0.5)
        
        # Calculate weighted matches for job keywords  
        for keyword, weight in job_keywords.items():
            if keyword in section_keywords:
                section_freq = section_keywords[keyword]
                job_score += min(weight * section_freq * 0.15, 0.6)
        
        # Combine scores (job keywords weighted higher)
        combined_score = (persona_score * 0.4) + (job_score * 0.6)
        
        return min(combined_score, 1.0)
    
    def _calculate_title_relevance(self, 
                                 title_text: str,
                                 persona: str,
                                 job: str,
                                 persona_keywords: Dict[str, float],
                                 job_keywords: Dict[str, float]) -> float:
        """Calculate title-specific relevance score"""
        
        score = 0.0
        
        # Direct keyword matches in title (high value)
        title_words = set(self.text_processor.extract_key_terms(title_text))
        
        for keyword in persona_keywords:
            if keyword in title_words:
                score += 0.3
                
        for keyword in job_keywords:
            if keyword in title_words:
                score += 0.4
        
        # Semantic similarity of title to queries
        title_persona_sim = self.text_processor.calculate_semantic_similarity(
            title_text, persona.lower()
        )
        title_job_sim = self.text_processor.calculate_semantic_similarity(
            title_text, job.lower()
        )
        
        score += (title_persona_sim * 0.2) + (title_job_sim * 0.3)
        
        # Special title patterns
        if self._has_methodology_indicators(title_text):
            score += 0.1
        if self._has_results_indicators(title_text):
            score += 0.1
            
        return min(score, 1.0)
    
    def _calculate_content_quality_score(self, section: DocumentSection) -> float:
        """Calculate content quality and structure score"""
        
        content = section.content.strip()
        content_length = len(content)
        word_count = section.word_count
        
        score = 0.0
        
        # Length scoring (prefer substantial but not excessive content)
        if 200 <= content_length <= 1500:
            score += 0.4
        elif 100 <= content_length < 200:
            score += 0.3
        elif 50 <= content_length < 100:
            score += 0.2
        elif content_length > 1500:
            score += 0.2  # Penalize overly long sections
        
        # Word count scoring
        if 30 <= word_count <= 300:
            score += 0.3
        elif word_count > 300:
            score += 0.2
            
        # Content structure indicators
        if self._has_structured_content(content):
            score += 0.2
            
        # Information density
        unique_words = len(set(content.lower().split()))
        if word_count > 0:
            diversity_ratio = unique_words / word_count
            if diversity_ratio > 0.6:  # Good vocabulary diversity
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_domain_specific_score(self, 
                                       section_text: str,
                                       persona: str,
                                       job: str) -> float:
        """Calculate domain-specific relevance bonuses"""
        
        score = 0.0
        
        # Academic/Research domain patterns
        if 'research' in persona.lower() or 'academic' in persona.lower():
            academic_indicators = [
                'methodology', 'experiment', 'hypothesis', 'literature',
                'analysis', 'findings', 'results', 'conclusion',
                'dataset', 'evaluation', 'benchmark'
            ]
            matches = sum(1 for indicator in academic_indicators 
                         if indicator in section_text)
            score += min(matches * 0.1, 0.3)
        
        # Business domain patterns
        if any(term in persona.lower() for term in ['analyst', 'manager', 'business']):
            business_indicators = [
                'strategy', 'market', 'revenue', 'profit', 'investment',
                'growth', 'competitive', 'financial', 'budget', 'roi'
            ]
            matches = sum(1 for indicator in business_indicators 
                         if indicator in section_text)
            score += min(matches * 0.1, 0.3)
        
        # Technical domain patterns
        if any(term in persona.lower() for term in ['developer', 'engineer', 'technical']):
            technical_indicators = [
                'algorithm', 'implementation', 'system', 'architecture',
                'framework', 'optimization', 'performance', 'design'
            ]
            matches = sum(1 for indicator in technical_indicators 
                         if indicator in section_text)
            score += min(matches * 0.1, 0.3)
        
        # Job-specific patterns
        job_lower = job.lower()
        if 'menu' in job_lower or 'food' in job_lower:
            food_indicators = [
                'ingredient', 'recipe', 'nutrition', 'dietary', 'cuisine',
                'cooking', 'preparation', 'meal', 'dish', 'flavor'
            ]
            matches = sum(1 for indicator in food_indicators 
                         if indicator in section_text)
            score += min(matches * 0.15, 0.4)
        
        return min(score, 1.0)
    
    def _has_methodology_indicators(self, text: str) -> bool:
        """Check if text contains methodology indicators"""
        methodology_terms = [
            'method', 'approach', 'technique', 'procedure',
            'process', 'framework', 'algorithm', 'implementation'
        ]
        return any(term in text for term in methodology_terms)
    
    def _has_results_indicators(self, text: str) -> bool:
        """Check if text contains results indicators"""
        results_terms = [
            'result', 'finding', 'outcome', 'conclusion',
            'analysis', 'evaluation', 'performance', 'summary'
        ]
        return any(term in text for term in results_terms)
    
    def _has_structured_content(self, content: str) -> bool:
        """Check if content shows good structure"""
        # Look for lists, numbered items, or clear paragraphs
        structure_indicators = [
            r'\d+\.',  # Numbered lists
            r'•',      # Bullet points
            r'-\s',    # Dash lists
            r'\n\s*\n', # Paragraph breaks
            r':\s*\n',  # Colons followed by newlines
            r'(first|second|third|finally|moreover|furthermore|however)'  # Transition words
        ]
        
        return any(re.search(pattern, content, re.IGNORECASE) 
                  for pattern in structure_indicators)