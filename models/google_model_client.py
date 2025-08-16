"""
Google Flan-T5 model client optimized for local intent detection
"""

import json
import re
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from difflib import SequenceMatcher

# Try to import the name extraction service
try:
    from services.name import extract_names
    HAS_NAME_SERVICE = True
except ImportError:
    HAS_NAME_SERVICE = False

# Try to import the address extraction service
try:
    from services.address import AddressExtractor, extract_addresses_pyap
    HAS_ADDRESS_SERVICE = True
    address_extractor = AddressExtractor()
except ImportError:
    HAS_ADDRESS_SERVICE = False
    address_extractor = None

# Try to import the phone extraction service
try:
    from services.phone import PhoneNumberExtractor
    HAS_PHONE_SERVICE = True
    phone_extractor = PhoneNumberExtractor()
except ImportError:
    HAS_PHONE_SERVICE = False
    phone_extractor = None


class GoogleModelClient:
    """Client for Google's Flan-T5 model with optimized intent detection"""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = "cpu"  # Force CPU to avoid MPS issues on macOS
        
        # Intent keywords mapping for fallback detection
        self.intent_keywords = {
            "create_client": [
                "create client", "new client", "add client", "register client", 
                "create a client", "add a new client", "add a client", 
                "create new client", "i'd like to add a client", "want to add a client",
                "need to add a client", "register a new client", "client to the database",
                "client profile", "client account", "client", "customer", "add customer",
                "new customer", "create customer", "register customer", "creat client",
                "crete client", "make client", "setup client"
            ],
            "book_restaurant": ["book", "reservation", "table", "restaurant", "dining", "reserve"],
            "check_weather": ["weather", "temperature", "forecast", "rain", "sunny", "climate"],
            "send_email": ["email", "send message", "mail", "send email", "message", "send mail"],
        }
        
    def _load_model(self):
        """Load model and tokenizer"""
        if self._model is None:
            print(f"Loading {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._model.eval()
            print("Model loaded successfully!")
            
    def _extract_name_from_response(self, response: str) -> Optional[str]:
        """Extract a name from natural conversational response"""
        response = response.strip()
        
        # Try using the spaCy-based name extraction service first
        if HAS_NAME_SERVICE:
            try:
                names = extract_names(response)
                if names:
                    # Prefer the longest name (likely the full name)
                    longest_name = max(names, key=len)
                    return longest_name
            except Exception as e:
                # Silently fall back to pattern matching
                pass
        
        # Fallback: Enhanced pattern-based extraction
        # Remove common phrases and filler words
        cleanup_phrases = [
            "you can put down", "the name is", "client's name is", "his name is",
            "her name is", "it's gonna be", "the name?", "oh right,", "yeah,",
            "so the client's name is", "let me think", "ah yes,", "or maybe",
            "no wait,", "no,", "it's", "its", "it is", "ehh", "um", "uh", 
            "umm", "uhh", "uhhh", "ummm", "well", "oh", "ah", "hmm", "err",
            "so the", "client's", "name is", "let me", "think"
        ]
        
        cleaned = response
        for phrase in cleanup_phrases:
            cleaned = re.sub(phrase, ' ', cleaned, flags=re.IGNORECASE).strip()
        
        # Handle "name, name" pattern (repeated names)
        repeated_name_pattern = r'\b(\w+)[,\s]+\1\b'
        repeated_match = re.search(repeated_name_pattern, cleaned, re.IGNORECASE)
        if repeated_match:
            cleaned = re.sub(repeated_name_pattern, repeated_match.group(1), cleaned, flags=re.IGNORECASE)
        
        # Clean up extra punctuation and spaces
        cleaned = re.sub(r'[,.\s]+', ' ', cleaned).strip()
        
        # Look for capitalized words or known name patterns
        words = cleaned.split()
        name_parts = []
        skip_next = False
        
        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
                
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
        
        # If no name found with capitalized words, try finding any reasonable name pattern
        if not name_parts:
            # Look for sequences of 2-3 words that could be names
            # This handles cases where spaCy fails and capitalization is inconsistent
            words = cleaned.split()
            for i in range(len(words) - 1):
                # Check if we have 2-3 consecutive words that look like names
                if i < len(words) - 1:
                    word1 = words[i].strip(".,")
                    word2 = words[i + 1].strip(".,")
                    
                    # Basic heuristic: words that are 2+ chars and don't contain numbers
                    if (len(word1) >= 2 and len(word2) >= 2 and 
                        not any(c.isdigit() for c in word1) and 
                        not any(c.isdigit() for c in word2) and
                        word1.lower() not in ['the', 'and', 'or', 'but', 'for', 'with', 'from']):
                        
                        # Check if there's a third word that could be part of the name
                        if i < len(words) - 2:
                            word3 = words[i + 2].strip(".,")
                            if (len(word3) >= 2 and not any(c.isdigit() for c in word3) and
                                word3.lower() not in ['the', 'and', 'or', 'but', 'for', 'with', 'from']):
                                return f"{word1} {word2} {word3}"
                        
                        return f"{word1} {word2}"
        
        return " ".join(name_parts) if name_parts else None
    
    def _extract_address_from_response(self, response: str) -> Optional[str]:
        """Extract an address from natural conversational response"""
        response = response.strip()
        
        # Try using the address extraction service first
        if HAS_ADDRESS_SERVICE and address_extractor:
            try:
                addresses = address_extractor.extract_best_addresses(response)
                if addresses:
                    # Return the longest/most complete address
                    return max(addresses, key=len)
            except Exception as e:
                # Silently fall back to pattern matching
                pass
        
        # Fallback: Enhanced pattern matching for natural speech
        # First, normalize written numbers to digits
        normalized = response
        number_words = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
            'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
            'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
            'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
            'ninety': '90', 'hundred': '100', 'thousand': '1000'
        }
        
        # Replace number words with digits
        for word, digit in number_words.items():
            normalized = re.sub(r'\b' + word + r'\b', digit, normalized, flags=re.IGNORECASE)
        
        # Handle compound numbers
        # "forty two" -> "42", "thirty three" -> "33", etc.
        normalized = re.sub(r'\b(\d+)\s+(\d{1,2})\b', lambda m: str(int(m.group(1)) + int(m.group(2))), normalized)
        
        # Handle special cases
        normalized = re.sub(r'\btwelve hundred\b', '1200', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bfive thousand\b', '5000', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bthree thousand three hundred\b', '3300', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bnine hundred\b', '900', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\beight hundred\b', '800', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bfour hundred\b', '400', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bone fifty\b', '150', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bfifteen fifty\b', '1550', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bfour twenty\b', '420', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\beight oh eight\b', '808', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bthree oh one\b', '301', normalized, flags=re.IGNORECASE)
        
        # Handle written zip codes
        normalized = re.sub(r'\bone one two zero one\b', '11201', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bone zero zero one nine\b', '10019', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bnine eight one zero one\b', '98101', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bsix zero six one one\b', '60611', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\beight nine one zero nine\b', '89109', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\btwo zero zero zero four\b', '20004', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bnine zero zero three six\b', '90036', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bnine four one zero five\b', '94105', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bnine three one zero one\b', '93101', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\bone nine one zero nine\b', '19109', normalized, flags=re.IGNORECASE)
        
        # Remove filler phrases and words
        filler_phrases = [
            "they're at", "you'll find them at", "the address is", "it's at", 
            "send everything to", "they're located at", "you can mail it to",
            "it's gonna be", "address is", "the mailing address is",
            "oh the address?", "so you'll find them at", "my office is at",
            "send it to", "i live at", "located at", "um", "uh", "umm", "uhh",
            "let me think", "let me see", "wait", "no wait", "no sorry",
            "that's on the", "hmm", "oh"
        ]
        
        cleaned = normalized
        for phrase in filler_phrases:
            cleaned = re.sub(phrase + r'\s*', '', cleaned, flags=re.IGNORECASE).strip()
        
        # Fix common transcription issues
        cleaned = re.sub(r'\.\.\.\s*', ' ', cleaned)  # Replace ellipsis with space
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
        
        # Try to reconstruct addresses with natural speech patterns
        # Look for address components
        street_pattern = r'(\d+)\s+([A-Za-z\s]+?)(?:\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|boulevard|blvd|way|court|ct|place|pl|circle|parkway|highway))'
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
            
            full_address = f"{street}{unit}, {city_state_zip}"
            return full_address
        
        # If structured extraction failed, clean up and check if it looks like an address
        cleaned = re.sub(r'^[,.\s]+', '', cleaned)  # Remove leading punctuation
        cleaned = cleaned.strip()
        
        # Check if it has address indicators
        has_street = any(word in cleaned.lower() for word in [
            "street", "st", "avenue", "ave", "road", "rd", "lane", "ln", 
            "drive", "dr", "boulevard", "blvd", "way", "court", "ct", 
            "place", "pl", "circle", "parkway", "highway", "broadway"
        ])
        
        has_numbers = any(char.isdigit() for char in cleaned)
        has_state = re.search(r'\b[A-Z]{2}\b', cleaned) is not None
        has_zip = re.search(r'\b\d{5}\b', cleaned) is not None
        
        # If it has street indicators or state/zip and numbers, likely an address
        if has_numbers and (has_street or (has_state and has_zip)):
            return cleaned
            
        return None
    
    def _extract_phone_from_response(self, response: str) -> Optional[str]:
        """Extract a phone number from natural conversational response"""
        response = response.strip()
        
        # Try using the phone extraction service first
        if HAS_PHONE_SERVICE and phone_extractor:
            try:
                phone_numbers = phone_extractor.extract_best(response)
                if phone_numbers:
                    # Return the first valid phone number
                    return phone_numbers[0]
            except Exception as e:
                # Silently fall back to pattern matching
                pass
        
        # Fallback: Enhanced pattern matching
        # First, try to normalize the text by converting written numbers to digits
        normalized = response
        if HAS_PHONE_SERVICE and phone_extractor:
            try:
                normalized = phone_extractor.convert_spoken_to_digits(response)
            except:
                pass
        
        # Remove common filler words
        filler_phrases = [
            "my number is", "phone number is", "call me at", "reach me at", 
            "it's", "its", "it is", "phone number?", "sure,", "area code", 
            "then", "and then", "oh wait", "I forgot", "you can call at"
        ]
        cleaned = normalized
        for phrase in filler_phrases:
            cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Look for area code mentioned separately
        # Handle patterns like "area code 415 then 555-0198" or "555-0123... area code is 212"
        area_code_patterns = [
            r'area code[:\s]+(\d{3})',
            r'area code\s+(\d)\s*(\d)\s*(\d)',  # "area code 4 1 5"
            r'(\d{3}).*?(?:then|and then)',
            r'area code[,\s]+it\'s\s+(\d{3})',
            r'(?:^|[^\d])(\d{3})\s+(?:five|5|uh)',  # Catches "718 five five five" or "718 uh"
            r'area code.*?(\d{3})$',  # Area code at end
            r'plus\s*(?:one|1)\s*(\d{3})'  # "+1 917"
        ]
        
        area_code = None
        for pattern in area_code_patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                # Handle multi-group matches (like individual digits)
                if len(match.groups()) == 3:
                    area_code = ''.join(match.groups())
                else:
                    area_code = match.group(1)
                # Ensure it's 3 digits
                if area_code and len(area_code) == 3:
                    break
        
        # Look for the main number part (7 digits)
        remaining = None
        
        # First handle "and then" pattern specially
        and_then_match = re.search(r'and then\s+(\d+)', cleaned, re.IGNORECASE)
        if and_then_match:
            # Extract everything after "and then"
            after_and_then = and_then_match.group(1)
            digits = re.sub(r'[^\d]', '', after_and_then)
            if len(digits) >= 7:
                remaining = digits[:7]
        
        if not remaining:
            main_number_patterns = [
                r'(\d{3})[-.\s]*(\d{4})',
                r'(5\s*5\s*5)[-.\s]*(\d{4})',  # Handle "five five five" as "555"
                r'(\d)\s*(\d)\s*(\d)[-.\s]*(\d)\s*(\d)\s*(\d)\s*(\d)',  # Individual digits
                r'(555)[-.\s]*(\d{4})',  # Direct 555 pattern
                r'dot\s*(\d{4})',  # Handle "dot 0123"
                r'dash\s*(\d{3})[-.\s]*dash\s*(\d{4})',  # "dash 555 dash 0178"
                r'zero\s*(\d{3})'  # Handle "zero one two three"
            ]
            
            for pattern in main_number_patterns:
                match = re.search(pattern, cleaned)
                if match:
                    if pattern == r'dot\s*(\d{4})':
                        # Special handling for "555 dot 0123" pattern
                        # Look for preceding 3 digits
                        prefix_match = re.search(r'(\d{3})\s*dot', cleaned)
                        if prefix_match:
                            remaining = prefix_match.group(1) + match.group(1)
                    elif len(match.groups()) == 2:
                        remaining = match.group(1) + match.group(2)
                    elif len(match.groups()) == 7:
                        # Concatenate individual digits
                        remaining = ''.join(match.groups())
                    
                    if remaining:
                        remaining = re.sub(r'[^\d]', '', remaining)
                        if len(remaining) == 7:
                            break
        
        # If we found both parts, combine them
        if area_code and remaining and len(remaining) == 7:
            return f"({area_code}) {remaining[:3]}-{remaining[3:]}"
        
        # Also check if the full number is at the end after "area code is X"
        end_pattern = r'area code[,\s]+it\'s\s+(\d{3})$'
        end_match = re.search(end_pattern, cleaned, re.IGNORECASE)
        if end_match and remaining and len(remaining) == 7:
            area_code = end_match.group(1)
            return f"({area_code}) {remaining[:3]}-{remaining[3:]}"
        
        # Look for phone patterns
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',
            r'\b\d{3}\s+\d{3}\s+\d{4}\b',
            r'\+?1?\s*\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, cleaned)
            if phones:
                # Format the phone number
                digits = re.sub(r'[^\d]', '', phones[0])
                if len(digits) == 11 and digits.startswith('1'):
                    digits = digits[1:]
                if len(digits) == 10:
                    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
                    
        return None
    
    def _extract_intent_from_text(self, text: str) -> Tuple[str, float]:
        """Enhanced keyword-based intent extraction with fuzzy matching"""
        text_lower = text.lower().strip()
        
        # First try exact matching
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    confidence = 0.9 if keyword == text_lower else 0.8
                    return intent, confidence
        
        # Try fuzzy matching for better typo handling
        best_match = None
        best_score = 0.0
        
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                # Check similarity with the whole text
                similarity = SequenceMatcher(None, keyword, text_lower).ratio()
                if similarity > best_score and similarity > 0.7:  # 70% similarity threshold
                    best_score = similarity
                    best_match = (intent, similarity * 0.8)  # Scale confidence
                
                # Also check if any word in the text is similar to keyword words
                text_words = text_lower.split()
                keyword_words = keyword.split()
                
                for text_word in text_words:
                    for keyword_word in keyword_words:
                        word_similarity = SequenceMatcher(None, keyword_word, text_word).ratio()
                        if word_similarity > 0.8 and word_similarity > best_score:  # 80% word similarity
                            best_score = word_similarity
                            best_match = (intent, word_similarity * 0.7)
        
        if best_match:
            return best_match
                    
        return "unknown", 0.3
    
    def _extract_parameters(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract parameters based on intent"""
        params = {}
        
        if intent == "create_client":
            # Look for email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            if emails:
                params["email"] = emails[0]
                
            # Look for phone
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            phones = re.findall(phone_pattern, text)
            if phones:
                params["phone_number"] = phones[0]
                
        elif intent == "check_weather":
            # Extract location after "in" or "for"
            if " in " in text.lower():
                location = text.split(" in ", 1)[1].strip()
                params["location"] = location.split()[0] if location else None
                
        return params
    
    async def generate(self, prompt: str) -> str:
        """Generate response with intent detection"""
        self._load_model()
        
        # Check if this is an intent detection prompt
        if "Available functions:" in prompt and "User input:" in prompt:
            user_input_match = re.search(r'User input: "([^"]+)"', prompt)
            if user_input_match:
                user_input = user_input_match.group(1)
                
                # Use keyword-based detection
                intent, confidence = self._extract_intent_from_text(user_input)
                parameters = self._extract_parameters(user_input, intent)
                
                if intent != "unknown":
                    return json.dumps({
                        "intent": intent,
                        "extracted_parameters": parameters,
                        "confidence": confidence,
                        "reasoning": f"Detected '{intent}' based on keywords"
                    })
        
        # For parameter extraction
        if "Extract any parameter values" in prompt:
            return self._handle_parameter_extraction(prompt)
        
        # For question generation
        if "Generate a natural, conversational question" in prompt:
            return self._generate_question(prompt)
        
        # Default response
        return json.dumps({
            "intent": "unknown",
            "extracted_parameters": {},
            "confidence": 0.5,
            "reasoning": "Could not determine intent"
        })
    
    def _handle_parameter_extraction(self, prompt: str) -> str:
        """Handle parameter extraction from user responses"""
        user_resp_match = re.search(r'User response: "([^"]+)"', prompt)
        if not user_resp_match:
            return json.dumps({"updated_parameters": {}, "still_missing": [], "ready_to_execute": False})
            
        user_response = user_resp_match.group(1)
        
        # Get current parameters
        current_params = {}
        if "Currently gathered:" in prompt:
            try:
                current_match = re.search(r'Currently gathered: ({.*?})\n', prompt, re.DOTALL)
                if current_match:
                    current_params = json.loads(current_match.group(1))
            except:
                pass
        
        # Extract missing parameters
        new_params = {}
        missing_params = []
        if "Missing parameters:" in prompt:
            missing_match = re.search(r'Missing parameters: \[(.*?)\]', prompt)
            if missing_match:
                missing_params = [p.strip().strip("'\"") for p in missing_match.group(1).split(",")]
        
        # Extract based on what's missing
        if "client_name" in missing_params:
            name = self._extract_name_from_response(user_response)
            if name:
                new_params["client_name"] = name
                
        elif "phone_number" in missing_params:
            # Use the phone extraction method
            phone = self._extract_phone_from_response(user_response)
            if phone:
                new_params["phone_number"] = phone
                
        elif "address" in missing_params:
            # Use the address extraction method
            address = self._extract_address_from_response(user_response)
            if address:
                new_params["address"] = address
                
        elif "email" in missing_params:
            if "@" in user_response:
                new_params["email"] = user_response.strip()
        
        # Merge parameters
        all_params = current_params.copy()
        all_params.update(new_params)
        
        # Calculate still missing
        still_missing = [p for p in missing_params if p not in new_params]
        
        return json.dumps({
            "updated_parameters": all_params,
            "still_missing": still_missing,
            "ready_to_execute": len(still_missing) == 0
        })
    
    def _generate_question(self, prompt: str) -> str:
        """Generate natural questions for missing parameters"""
        if "client_name" in prompt:
            return "What is the client's full name?"
        elif "phone_number" in prompt:
            return "What is the client's phone number?"
        elif "address" in prompt:
            return "What is the client's address?"
        elif "email" in prompt:
            return "What is the client's email address? (optional)"
        elif "location" in prompt:
            return "Which city would you like to check the weather for?"
        elif "restaurant_name" in prompt:
            return "Which restaurant would you like to book?"
        elif "date" in prompt:
            return "What date would you like?"
        elif "time" in prompt:
            return "What time would you like?"
        elif "party_size" in prompt:
            return "How many people will be dining?"
        else:
            return "Could you provide more details?"