import spacy
from typing import List

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