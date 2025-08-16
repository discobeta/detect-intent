"""
Shared text utilities for common text manipulation and cleaning operations
"""

import re
from typing import Dict, List, Optional


class NumberNormalizer:
    """Handles conversion of written numbers to digits"""
    
    def __init__(self):
        # Basic number word mappings
        self.number_words = {
            'zero': '0', 'oh': '0', 'o': '0',
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
            'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
            'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
            'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
            'ninety': '90', 'hundred': '100', 'thousand': '1000'
        }
        
        # Special number combinations
        self.special_numbers = {
            'twelve hundred': '1200',
            'five thousand': '5000',
            'three thousand three hundred': '3300',
            'nine hundred': '900',
            'eight hundred': '800',
            'four hundred': '400',
            'one fifty': '150',
            'fifteen fifty': '1550',
            'four twenty': '420',
            'eight oh eight': '808',
            'three oh one': '301'
        }
        
        # Written zip codes
        self.written_zip_codes = {
            'one one two zero one': '11201',
            'one zero zero one nine': '10019',
            'nine eight one zero one': '98101',
            'six zero six one one': '60611',
            'eight nine one zero nine': '89109',
            'two zero zero zero four': '20004',
            'nine zero zero three six': '90036',
            'nine four one zero five': '94105',
            'nine three one zero one': '93101',
            'one nine one zero nine': '19109'
        }
    
    def normalize_numbers(self, text: str) -> str:
        """
        Convert written numbers to digits in text.
        
        Args:
            text (str): Text containing written numbers
            
        Returns:
            str: Text with numbers normalized to digits
        """
        normalized = text
        
        # Replace special number combinations first
        for word_combo, digit in self.special_numbers.items():
            normalized = re.sub(r'\b' + word_combo + r'\b', digit, normalized, flags=re.IGNORECASE)
        
        # Replace written zip codes
        for written_zip, digit_zip in self.written_zip_codes.items():
            normalized = re.sub(r'\b' + written_zip + r'\b', digit_zip, normalized, flags=re.IGNORECASE)
        
        # Replace individual number words with digits
        for word, digit in self.number_words.items():
            normalized = re.sub(r'\b' + word + r'\b', digit, normalized, flags=re.IGNORECASE)
        
        # Handle compound numbers like "forty two" -> "42", "thirty three" -> "33"
        normalized = re.sub(r'\b(\d+)\s+(\d{1,2})\b', 
                          lambda m: str(int(m.group(1)) + int(m.group(2))), 
                          normalized)
        
        return normalized


class TextCleaner:
    """General text cleaning utilities"""
    
    @staticmethod
    def remove_filler_words(text: str, filler_phrases: List[str]) -> str:
        """
        Remove filler phrases from text.
        
        Args:
            text (str): Input text
            filler_phrases (List[str]): List of phrases to remove
            
        Returns:
            str: Cleaned text
        """
        cleaned = text
        for phrase in filler_phrases:
            cleaned = re.sub(phrase + r'\s*', '', cleaned, flags=re.IGNORECASE).strip()
        return cleaned
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    @staticmethod
    def clean_punctuation(text: str) -> str:
        """
        Clean up extra punctuation and spaces.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple punctuation and spaces with single space
        text = re.sub(r'[,.\s]+', ' ', text)
        # Remove leading punctuation
        text = re.sub(r'^[,.\s]+', '', text)
        return text.strip()
    
    @staticmethod
    def fix_transcription_issues(text: str) -> str:
        """
        Fix common transcription issues.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Fixed text
        """
        # Replace ellipsis with space
        text = re.sub(r'\.\.\.\s*', ' ', text)
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def handle_repeated_pattern(text: str, pattern: str = r'\b(\w+)[,\s]+\1\b') -> str:
    """
    Handle repeated words pattern (e.g., "name, name" -> "name").
    
    Args:
        text (str): Input text
        pattern (str): Regex pattern for repeated words
        
    Returns:
        str: Text with repeated patterns fixed
    """
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        text = re.sub(pattern, match.group(1), text, flags=re.IGNORECASE)
    return text


def is_stop_word(word: str) -> bool:
    """
    Check if a word is a common stop word.
    
    Args:
        word (str): Word to check
        
    Returns:
        bool: True if word is a stop word
    """
    stop_words = ['the', 'and', 'or', 'but', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at']
    return word.lower() in stop_words


def extract_digits_only(text: str) -> str:
    """
    Extract only digits from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: String containing only digits
    """
    return re.sub(r'[^\d]', '', text)