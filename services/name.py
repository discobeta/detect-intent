import re
import spacy
from typing import List, Optional
from .text_utils import TextCleaner, handle_repeated_pattern, is_stop_word


class NameExtractor:
    """Service for extracting and cleaning names from text"""
    
    def __init__(self):
        # Name-specific filler phrases
        self.name_filler_phrases = [
            "you can put down", "the name is", "client's name is", "his name is",
            "her name is", "it's gonna be", "the name?", "oh right,", "yeah,",
            "so the client's name is", "let me think", "ah yes,", "or maybe",
            "no wait,", "no,", "it's", "its", "it is", "ehh", "um", "uh", 
            "umm", "uhh", "uhhh", "ummm", "well", "oh", "ah", "hmm", "err",
            "so the", "client's", "name is", "let me", "think"
        ]
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("Warning: spaCy model not available for name extraction")
    
    def clean_name_response(self, response: str) -> str:
        """
        Clean a response containing a name by removing filler phrases.
        
        Args:
            response (str): Raw response text
            
        Returns:
            str: Cleaned response
        """
        cleaned = response.strip()
        
        # Remove filler phrases
        cleaned = TextCleaner.remove_filler_words(cleaned, self.name_filler_phrases)
        
        # Handle repeated names pattern (e.g., "raj, rajesh patel" -> "rajesh patel")
        cleaned = handle_repeated_pattern(cleaned)
        
        # Clean up punctuation and normalize whitespace
        cleaned = TextCleaner.clean_punctuation(cleaned)
        
        return cleaned
    
    def extract_name_parts(self, cleaned_text: str) -> List[str]:
        """
        Extract potential name parts from cleaned text.
        
        Args:
            cleaned_text (str): Cleaned text
            
        Returns:
            List[str]: List of potential name parts
        """
        words = cleaned_text.split()
        name_parts = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\'-]', '', word)
            
            # Check if it's a name component
            if clean_word and len(clean_word) > 1:
                # Accept capitalized words or words with apostrophes (O'Brien, etc.)
                if clean_word[0].isupper() or "'" in clean_word:
                    name_parts.append(clean_word)
                # Handle lowercase name components (van, der, de, etc.)
                elif clean_word.lower() in ['van', 'der', 'de', 'la', 'von', 'del', 'di', 'da']:
                    name_parts.append(clean_word)
                # Handle titles
                elif clean_word.lower() in ['dr', 'mr', 'mrs', 'ms', 'prof', 'professor']:
                    name_parts.append(clean_word.capitalize() + '.')
        
        # Handle special cases like "raj, rajesh patel" -> "rajesh patel"
        if len(name_parts) >= 2:
            # If first name is substring of second, use the longer one
            if name_parts[0].lower() in name_parts[1].lower():
                name_parts.pop(0)
        
        return name_parts
    
    def find_name_in_uncapitalized_text(self, cleaned_text: str) -> Optional[str]:
        """
        Try to find names in text that may not be properly capitalized.
        
        Args:
            cleaned_text (str): Cleaned text
            
        Returns:
            Optional[str]: Found name or None
        """
        words = cleaned_text.split()
        
        for i in range(len(words) - 1):
            if i < len(words) - 1:
                word1 = words[i].strip(".,")
                word2 = words[i + 1].strip(".,")
                
                # Basic heuristic: words that are 2+ chars and don't contain numbers
                if (len(word1) >= 2 and len(word2) >= 2 and 
                    not any(c.isdigit() for c in word1) and 
                    not any(c.isdigit() for c in word2) and
                    not is_stop_word(word1) and not is_stop_word(word2)):
                    
                    # Check if there's a third word that could be part of the name
                    if i < len(words) - 2:
                        word3 = words[i + 2].strip(".,")
                        if (len(word3) >= 2 and not any(c.isdigit() for c in word3) and
                            not is_stop_word(word3)):
                            return f"{word1} {word2} {word3}"
                    
                    return f"{word1} {word2}"
        
        return None
    
    def extract_from_response(self, response: str) -> Optional[str]:
        """
        Extract a name from natural conversational response.
        
        Args:
            response (str): Raw response text
            
        Returns:
            Optional[str]: Extracted name or None
        """
        response = response.strip()
        
        # Try using spaCy first if available
        if self.nlp:
            try:
                names = extract_names(response)
                if names:
                    # Prefer the longest name (likely the full name)
                    return max(names, key=len)
            except:
                pass
        
        # Clean the response
        cleaned = self.clean_name_response(response)
        
        # Extract name parts
        name_parts = self.extract_name_parts(cleaned)
        
        # If no capitalized names found, try uncapitalized
        if not name_parts:
            name = self.find_name_in_uncapitalized_text(cleaned)
            if name:
                return name
        
        return " ".join(name_parts) if name_parts else None


def extract_names(text: str) -> List[str]:
    """
    Extract person names from text using spaCy NER.
    
    Args:
        text (str): The input text to extract names from
        
    Returns:
        List[str]: A list of extracted person names
        
    Example:
        >>> extract_names("John Smith and Mary Johnson work at Google with Bob Wilson.")
        ['John Smith', 'Mary Johnson', 'Bob Wilson']
    """
    try:
        # Load the English language model
        nlp = spacy.load("en_core_web_sm")
        
        # Process the text
        doc = nlp(text)
        
        # Extract person names (entities labeled as 'PERSON')
        names = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        
        # Remove duplicates while preserving order
        unique_names = list(dict.fromkeys(names))
        print("Extracted names: ", unique_names)
        return unique_names
        
    except OSError:
        raise Exception("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")
    except Exception as e:
        raise Exception(f"Error processing text: {str(e)}")