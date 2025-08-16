import re
from typing import List, Optional, Tuple, Dict
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat

def convert_words_to_digits(text: str) -> str:
    """
    Convert written number words to digits in text.
    Handles both individual digits and phone-number-like sequences.
    
    Args:
        text (str): Text containing written numbers
        
    Returns:
        str: Text with number words converted to digits
        
    Example:
        >>> convert_words_to_digits("seven one 8 92 four 71 88")
        "7 1 8 92 4 71 88"
        >>> convert_words_to_digits("call me at five five five one two three four five six seven")
        "call me at 5 5 5 1 2 3 4 5 6 7"
    """
    # Mapping of word numbers to digits
    word_to_digit = {
        'zero': '0', 'oh': '0', 'o': '0',
        'one': '1', 'won': '1',
        'two': '2', 'to': '2', 'too': '2',
        'three': '3', 'tree': '3',
        'four': '4', 'for': '4', 'fore': '4',
        'five': '5',
        'six': '6', 'sicks': '6',
        'seven': '7',
        'eight': '8', 'ate': '8',
        'nine': '9', 'niner': '9'
    }
    
    # Create pattern that matches any of the word numbers
    word_pattern = r'\b(' + '|'.join(word_to_digit.keys()) + r')\b'
    
    def replace_word(match):
        word = match.group(1).lower()
        return word_to_digit.get(word, word)
    
    # Convert words to digits
    result = re.sub(word_pattern, replace_word, text, flags=re.IGNORECASE)
    return result


def extract_phone_from_spoken_text(text: str) -> List[str]:
    """
    Extract phone numbers from text that may contain spoken/written numbers.
    
    Args:
        text (str): Text with potentially spoken phone numbers
        
    Returns:
        List[str]: List of extracted phone number digit sequences
        
    Example:
        >>> extract_phone_from_spoken_text("seven one 8 92 four 71 88")
        ['7189247188']
    """
    # First convert words to digits
    converted_text = convert_words_to_digits(text)
    
    # Look for sequences that could be phone numbers
    # Pattern: sequences of digits and spaces that could total 10-11 digits
    potential_sequences = []
    
    # Split into words and filter for digits and potential phone sequences
    words = converted_text.split()
    
    # Look for sequences of numbers that could form a phone number
    current_sequence = []
    digit_count = 0
    
    for word in words:
        # Check if word contains digits
        digits_in_word = re.sub(r'[^\d]', '', word)
        
        if digits_in_word:
            current_sequence.append(digits_in_word)
            digit_count += len(digits_in_word)
            
            # If we have enough digits for a phone number, check if it's complete
            if digit_count >= 10:
                phone_candidate = ''.join(current_sequence)
                
                # Take first 10-11 digits
                if digit_count >= 11:
                    phone_candidate = phone_candidate[:11]
                elif digit_count == 10:
                    phone_candidate = phone_candidate[:10]
                
                potential_sequences.append(phone_candidate)
                
                # Reset for next potential number
                current_sequence = []
                digit_count = 0
        else:
            # Non-digit word breaks the sequence
            if digit_count >= 7:  # Save partial sequences that might be phone numbers
                phone_candidate = ''.join(current_sequence)
                if len(phone_candidate) >= 7:  # At least area code + some digits
                    potential_sequences.append(phone_candidate)
            
            current_sequence = []
            digit_count = 0
    
    # Check final sequence
    if digit_count >= 7:
        phone_candidate = ''.join(current_sequence)
        potential_sequences.append(phone_candidate)
    
    return potential_sequences


def extract_and_format_phone_numbers(text: str, default_region: str = 'US') -> List[str]:
    """
    Extract phone numbers from text and format them to US standard format.
    Uses the phonenumbers library for accurate parsing and validation.
    Also handles spoken/written numbers.
    
    Args:
        text (str): Natural language text containing phone numbers
        default_region (str): Default country code for parsing
        
    Returns:
        List[str]: List of formatted US phone numbers in (XXX) XXX-XXXX format
        
    Example:
        >>> text = "Call me at 555-123-4567 or seven one 8 nine two four seven one eight eight"
        >>> extract_and_format_phone_numbers(text)
        ['(555) 123-4567', '(718) 924-7188']
    """
    phone_numbers = []
    
    # Method 1: Standard phone number extraction
    for match in phonenumbers.PhoneNumberMatcher(text, default_region):
        phone_number = match.number
        
        # Validate that it's a valid US number
        if (phonenumbers.is_valid_number(phone_number) and 
            phonenumbers.region_code_for_number(phone_number) == 'US'):
            
            # Format to US standard format
            formatted = phonenumbers.format_number(phone_number, PhoneNumberFormat.NATIONAL)
            phone_numbers.append(formatted)
    
    # Method 2: Handle spoken/written numbers
    spoken_candidates = extract_phone_from_spoken_text(text)
    for candidate in spoken_candidates:
        try:
            # Try to parse as phone number
            parsed = phonenumbers.parse(candidate, default_region)
            if (phonenumbers.is_valid_number(parsed) and 
                phonenumbers.region_code_for_number(parsed) == 'US'):
                formatted = phonenumbers.format_number(parsed, PhoneNumberFormat.NATIONAL)
                phone_numbers.append(formatted)
        except:
            continue
    
    return list(set(phone_numbers))  # Remove duplicates


def extract_phone_numbers_regex(text: str) -> List[str]:
    """
    Extract phone numbers using regex patterns (fallback method).
    
    Args:
        text (str): Text containing phone numbers
        
    Returns:
        List[str]: List of raw phone number strings found
    """
    # Common phone number patterns
    patterns = [
        r'\b\d{3}-\d{3}-\d{4}\b',                          # 555-123-4567
        r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',                    # (555) 123-4567
        r'\b\(\d{3}\)\s*\d{3}\s*\d{4}\b',                  # (555) 123 4567
        r'\b\d{3}\s+\d{3}\s+\d{4}\b',                      # 555 123 4567
        r'\b\d{3}\.\d{3}\.\d{4}\b',                        # 555.123.4567
        r'\b\d{10}\b',                                      # 5551234567
        r'\b1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',      # 1-555-123-4567, 1.555.123.4567
        r'\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',    # +1-555-123-4567
    ]
    
    phone_numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phone_numbers.extend(matches)
    
    return list(set(phone_numbers))  # Remove duplicates


def clean_phone_number(phone_str: str) -> str:
    """
    Clean a phone number string by removing non-digit characters.
    
    Args:
        phone_str (str): Raw phone number string
        
    Returns:
        str: Cleaned phone number with only digits
    """
    return re.sub(r'[^\d]', '', phone_str)


def format_us_phone_number(phone_str: str) -> Optional[str]:
    """
    Format a phone number string to US standard format (XXX) XXX-XXXX.
    
    Args:
        phone_str (str): Phone number string (can be messy)
        
    Returns:
        Optional[str]: Formatted phone number or None if invalid
        
    Example:
        >>> format_us_phone_number("5551234567")
        '(555) 123-4567'
        >>> format_us_phone_number("1-555-123-4567")
        '(555) 123-4567'
    """
    # Clean the number
    digits = clean_phone_number(phone_str)
    
    # Handle different digit lengths
    if len(digits) == 11 and digits.startswith('1'):
        # Remove leading 1 for US numbers
        digits = digits[1:]
    elif len(digits) == 10:
        # Perfect length for US number
        pass
    else:
        # Invalid length
        return None
    
    # Validate US phone number format
    if len(digits) != 10:
        return None
    
    # Basic validation for US numbers
    area_code = digits[:3]
    exchange = digits[3:6]
    
    # Area code can't start with 0 or 1
    if area_code[0] in ['0', '1']:
        return None
    
    # Exchange can't start with 0 or 1
    if exchange[0] in ['0', '1']:
        return None
    
    # Format to (XXX) XXX-XXXX
    return f"({area_code}) {exchange}-{digits[6:]}"


def extract_and_format_phone_regex(text: str) -> List[str]:
    """
    Extract and format phone numbers using regex approach (no external dependencies).
    Also handles spoken/written numbers.
    
    Args:
        text (str): Text containing phone numbers
        
    Returns:
        List[str]: List of formatted US phone numbers
    """
    formatted_numbers = []
    
    # Method 1: Extract regular formatted numbers
    raw_numbers = extract_phone_numbers_regex(text)
    for number in raw_numbers:
        formatted = format_us_phone_number(number)
        if formatted:
            formatted_numbers.append(formatted)
    
    # Method 2: Handle spoken numbers
    spoken_candidates = extract_phone_from_spoken_text(text)
    for candidate in spoken_candidates:
        formatted = format_us_phone_number(candidate)
        if formatted:
            formatted_numbers.append(formatted)
    
    return list(set(formatted_numbers))  # Remove duplicates


def validate_us_phone_number(phone_str: str) -> bool:
    """
    Validate if a string represents a valid US phone number.
    
    Args:
        phone_str (str): Phone number string
        
    Returns:
        bool: True if valid US phone number
    """
    try:
        # Try with phonenumbers library first
        parsed = phonenumbers.parse(phone_str, 'US')
        return (phonenumbers.is_valid_number(parsed) and 
                phonenumbers.region_code_for_number(parsed) == 'US')
    except:
        # Fall back to basic validation
        digits = clean_phone_number(phone_str)
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        
        if len(digits) != 10:
            return False
        
        # Basic US phone number rules
        area_code = digits[:3]
        exchange = digits[3:6]
        
        return not (area_code[0] in ['0', '1'] or exchange[0] in ['0', '1'])


class PhoneNumberExtractor:
    """
    A comprehensive phone number extractor with multiple methods.
    Handles regular formatted numbers and spoken/written numbers.
    """
    
    def __init__(self, default_region: str = 'US'):
        self.default_region = default_region
    
    def extract_from_spoken_text(self, text: str) -> List[str]:
        """
        Extract phone numbers specifically from spoken/written text.
        
        Args:
            text (str): Text with spoken numbers like "seven one eight nine two four seven one eight eight"
            
        Returns:
            List[str]: Formatted phone numbers
            
        Example:
            >>> extractor = PhoneNumberExtractor()
            >>> extractor.extract_from_spoken_text("seven one 8 92 four 71 88")
            ['(718) 924-7188']
        """
        spoken_candidates = extract_phone_from_spoken_text(text)
        formatted_numbers = []
        
        for candidate in spoken_candidates:
            # Try with phonenumbers library first
            try:
                parsed = phonenumbers.parse(candidate, self.default_region)
                if (phonenumbers.is_valid_number(parsed) and 
                    phonenumbers.region_code_for_number(parsed) == 'US'):
                    formatted = phonenumbers.format_number(parsed, PhoneNumberFormat.NATIONAL)
                    formatted_numbers.append(formatted)
                    continue
            except:
                pass
            
            # Fall back to regex formatting
            formatted = format_us_phone_number(candidate)
            if formatted:
                formatted_numbers.append(formatted)
        
        return list(set(formatted_numbers))
    
    def convert_spoken_to_digits(self, text: str) -> str:
        """
        Convert spoken number words to digits.
        
        Args:
            text (str): Text with spoken numbers
            
        Returns:
            str: Text with numbers converted to digits
        """
        return convert_words_to_digits(text)
    
    def extract_all_methods(self, text: str) -> dict:
        """
        Extract phone numbers using all available methods.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Results from different extraction methods
        """
        results = {
            'phonenumbers_lib': [],
            'regex_method': [],
            'spoken_numbers': [],
            'best_formatted': []
        }
        
        # Method 1: phonenumbers library
        try:
            results['phonenumbers_lib'] = extract_and_format_phone_numbers(text, self.default_region)
        except Exception as e:
            print(f"phonenumbers library error: {e}")
        
        # Method 2: regex approach
        results['regex_method'] = extract_and_format_phone_regex(text)
        
        # Method 3: spoken numbers specifically
        results['spoken_numbers'] = self.extract_from_spoken_text(text)
        
        # Combine and deduplicate results
        all_numbers = set(results['phonenumbers_lib'] + results['regex_method'] + results['spoken_numbers'])
        results['best_formatted'] = list(all_numbers)
        
        return results
    
    def extract_best(self, text: str) -> List[str]:
        """
        Extract phone numbers using the best available method.
        Handles both regular and spoken number formats.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Best extracted and formatted phone numbers
        """
        try:
            # Try phonenumbers library first (most accurate)
            return extract_and_format_phone_numbers(text, self.default_region)
        except:
            # Fall back to regex method (includes spoken number handling)
            return extract_and_format_phone_regex(text)
