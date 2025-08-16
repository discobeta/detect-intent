import re
import spacy
from typing import List, Dict, Optional
import usaddress
import pyap
from .text_utils import TextCleaner, NumberNormalizer

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
        
        # Initialize utilities
        self.number_normalizer = NumberNormalizer()
        
        # Address-specific filler phrases
        self.address_filler_phrases = [
            "they're at", "you'll find them at", "the address is", "it's at", 
            "send everything to", "they're located at", "you can mail it to",
            "it's gonna be", "address is", "the mailing address is",
            "oh the address?", "so you'll find them at", "my office is at",
            "send it to", "i live at", "located at", "um", "uh", "umm", "uhh",
            "let me think", "let me see", "wait", "no wait", "no sorry",
            "that's on the", "hmm", "oh"
        ]
        
        # Common street types
        self.street_types = [
            "street", "st", "avenue", "ave", "road", "rd", "lane", "ln", 
            "drive", "dr", "boulevard", "blvd", "way", "court", "ct", 
            "place", "pl", "circle", "parkway", "highway", "broadway"
        ]
    
    def clean_address_response(self, response: str) -> str:
        """
        Clean a response containing an address by removing filler phrases.
        
        Args:
            response (str): Raw response text
            
        Returns:
            str: Cleaned response
        """
        cleaned = response.strip()
        
        # Remove filler phrases
        cleaned = TextCleaner.remove_filler_words(cleaned, self.address_filler_phrases)
        
        # Fix common transcription issues
        cleaned = TextCleaner.fix_transcription_issues(cleaned)
        
        # Remove leading punctuation
        cleaned = re.sub(r'^[,.\s]+', '', cleaned)
        
        return cleaned.strip()
    
    def normalize_address_numbers(self, text: str) -> str:
        """
        Normalize written numbers in addresses to digits.
        
        Args:
            text (str): Text with potential written numbers
            
        Returns:
            str: Text with normalized numbers
        """
        return self.number_normalizer.normalize_numbers(text)
    
    def extract_from_response(self, response: str) -> Optional[str]:
        """
        Extract an address from natural conversational response.
        
        Args:
            response (str): Raw response text
            
        Returns:
            Optional[str]: Extracted address or None
        """
        response = response.strip()
        
        # Try using pyap first (most accurate)
        try:
            addresses = extract_addresses_pyap(response)
            if addresses:
                # Return the longest/most complete address
                return max(addresses, key=len)
        except:
            pass
        
        # Normalize numbers and clean the response
        normalized = self.normalize_address_numbers(response)
        cleaned = self.clean_address_response(normalized)
        
        # Try to reconstruct addresses with natural speech patterns
        address = self._reconstruct_address(cleaned)
        if address:
            return address
        
        # Check if the cleaned text looks like an address
        if self._looks_like_address(cleaned):
            return cleaned
        
        return None
    
    def _reconstruct_address(self, cleaned: str) -> Optional[str]:
        """
        Try to reconstruct an address from cleaned text.
        
        Args:
            cleaned (str): Cleaned text
            
        Returns:
            Optional[str]: Reconstructed address or None
        """
        # Look for address components
        street_pattern = r'(\d+)\s+([A-Za-z\s]+?)(?:\s+(?:' + '|'.join(self.street_types) + '))'
        unit_pattern = r'(?:apartment|apt|suite|ste|unit|floor|#)\s*([A-Za-z0-9]+)'
        city_state_pattern = r'([A-Za-z\s]+?)[\s,]+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)'
        
        # Try to find street address
        street_match = re.search(street_pattern, cleaned, re.IGNORECASE)
        unit_match = re.search(unit_pattern, cleaned, re.IGNORECASE)
        city_state_match = re.search(city_state_pattern, cleaned)
        
        if street_match and city_state_match:
            # Reconstruct the address
            street = street_match.group(0)
            unit = f", {unit_match.group(0)}" if unit_match else ""
            city_state_zip = city_state_match.group(0)
            
            return f"{street}{unit}, {city_state_zip}"
        
        return None
    
    def _looks_like_address(self, text: str) -> bool:
        """
        Check if text looks like it contains an address.
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text likely contains an address
        """
        # Check if it has street indicators
        has_street = any(word in text.lower() for word in self.street_types)
        
        # Check for numbers
        has_numbers = any(char.isdigit() for char in text)
        
        # Check for state abbreviation
        has_state = re.search(r'\b[A-Z]{2}\b', text) is not None
        
        # Check for ZIP code
        has_zip = re.search(r'\b\d{5}\b', text) is not None
        
        # If it has street indicators or state/zip and numbers, likely an address
        return has_numbers and (has_street or (has_state and has_zip))
    
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
