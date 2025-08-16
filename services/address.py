import re
import spacy
from typing import List, Dict, Optional
import usaddress
import pyap

def extract_addresses_pyap(text: str, country: str = 'US') -> List[str]:
    """
    Extract addresses using pyap library (recommended for most use cases).
    
    Args:
        text (str): Raw text containing addresses
        country (str): Country code ('US', 'CA', 'GB', etc.)
        
    Returns:
        List[str]: List of extracted addresses
        
    Example:
        >>> text = "Please send the package to 123 Main St, New York, NY 10001"
        >>> extract_addresses_pyap(text)
        ['123 Main St, New York, NY 10001']
    """
    try:
        addresses = pyap.parse(text, country=country)
        return [str(addr) for addr in addresses]
    except Exception as e:
        print(f"Error with pyap: {e}")
        return []


def extract_addresses_usaddress(text: str) -> List[Dict[str, str]]:
    """
    Extract and parse US addresses using usaddress library.
    Returns structured address components.
    
    Args:
        text (str): Raw text containing addresses
        
    Returns:
        List[Dict]: List of parsed address components
        
    Example:
        >>> extract_addresses_usaddress("123 Main St, New York, NY 10001")
        [{'AddressNumber': '123', 'StreetName': 'Main', 'StreetNamePostType': 'St', 
          'PlaceName': 'New York', 'StateName': 'NY', 'ZipCode': '10001'}]
    """
    # First, find potential addresses using regex
    potential_addresses = find_potential_addresses(text)
    
    parsed_addresses = []
    for addr in potential_addresses:
        try:
            parsed, addr_type = usaddress.tag(addr)
            if addr_type in ['Street Address', 'Ambiguous']:
                parsed_addresses.append(parsed)
        except Exception as e:
            print(f"Error parsing address '{addr}': {e}")
            continue
    
    return parsed_addresses


def extract_addresses_spacy(text: str) -> List[str]:
    """
    Extract addresses using spaCy NER (looks for location entities).
    Less accurate but doesn't require additional libraries.
    
    Args:
        text (str): Raw text containing addresses
        
    Returns:
        List[str]: List of potential addresses
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # Look for location entities and nearby text that might be addresses
        addresses = []
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC', 'FAC']:  # Geographic, Location, Facility
                # Get surrounding context
                start = max(0, ent.start - 5)
                end = min(len(doc), ent.end + 5)
                context = doc[start:end].text
                
                # Check if it looks like an address
                if is_likely_address(context):
                    addresses.append(context.strip())
        
        return list(set(addresses))  # Remove duplicates
        
    except OSError:
        raise Exception("spaCy English model not found. Please install it with: python -m spacy download en_core_web_sm")


def find_potential_addresses(text: str) -> List[str]:
    """
    Use regex patterns to find potential addresses in text.
    
    Args:
        text (str): Raw text
        
    Returns:
        List[str]: List of potential address strings
    """
    # Common address patterns
    patterns = [
        # Pattern 1: Number + Street + City, State ZIP
        r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl)[\s,]+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
        
        # Pattern 2: PO Box addresses
        r'\bP\.?O\.?\s*Box\s+\d+[\s,]+[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
        
        # Pattern 3: More flexible pattern
        r'\b\d+\s+[A-Za-z0-9\s\.,#-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl)[A-Za-z0-9\s\.,#-]*,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
    ]
    
    addresses = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        addresses.extend(matches)
    
    return list(set(addresses))  # Remove duplicates


def is_likely_address(text: str) -> bool:
    """
    Check if text looks like an address using heuristics.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text likely contains an address
    """
    # Check for common address indicators
    address_indicators = [
        r'\b\d+\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
        r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
        r'\b[A-Z]{2}\s+\d{5}\b',  # State + ZIP
        r'\bP\.?O\.?\s*Box\s+\d+\b'  # PO Box
    ]
    
    for pattern in address_indicators:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def extract_addresses_comprehensive(text: str) -> Dict[str, List]:
    """
    Extract addresses using multiple methods and return combined results.
    
    Args:
        text (str): Raw text containing addresses
        
    Returns:
        Dict: Results from different extraction methods
    """
    results = {
        'pyap_addresses': [],
        'usaddress_parsed': [],
        'spacy_addresses': [],
        'regex_addresses': []
    }
    
    # Method 1: pyap
    try:
        results['pyap_addresses'] = extract_addresses_pyap(text)
    except:
        pass
    
    # Method 2: usaddress
    try:
        results['usaddress_parsed'] = extract_addresses_usaddress(text)
    except:
        pass
    
    # Method 3: spaCy
    try:
        results['spacy_addresses'] = extract_addresses_spacy(text)
    except:
        pass
    
    # Method 4: Regex
    results['regex_addresses'] = find_potential_addresses(text)
    
    return results


class AddressExtractor:
    """
    A comprehensive address extractor class with multiple methods.
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
            print("Warning: spaCy model not available")
    
    def extract_best_addresses(self, text: str) -> List[str]:
        """
        Extract addresses using the best available method.
        
        Args:
            text (str): Raw text
            
        Returns:
            List[str]: Best extracted addresses
        """
        # Try pyap first (usually most accurate)
        try:
            addresses = extract_addresses_pyap(text)
            if addresses:
                return addresses
        except:
            pass
        
        # Fall back to regex patterns
        addresses = find_potential_addresses(text)
        if addresses:
            return addresses
        
        # Last resort: spaCy
        if self.nlp:
            return extract_addresses_spacy(text)
        
        return []
