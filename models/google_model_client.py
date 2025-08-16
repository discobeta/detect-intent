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
    from services.name import NameExtractor
    HAS_NAME_SERVICE = True
    name_extractor = NameExtractor()
except ImportError:
    HAS_NAME_SERVICE = False
    name_extractor = None

# Try to import the address extraction service
try:
    from services.address import AddressExtractor
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
        if HAS_NAME_SERVICE and name_extractor:
            return name_extractor.extract_from_response(response)
        return None
    
    def _extract_address_from_response(self, response: str) -> Optional[str]:
        """Extract an address from natural conversational response"""
        if HAS_ADDRESS_SERVICE and address_extractor:
            return address_extractor.extract_from_response(response)
        return None
    
    def _extract_phone_from_response(self, response: str) -> Optional[str]:
        """Extract a phone number from natural conversational response"""
        if HAS_PHONE_SERVICE and phone_extractor:
            return phone_extractor.extract_from_response(response)
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