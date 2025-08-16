#!/usr/bin/env python3
"""
Comprehensive test for agent client creation with various prompts and inputs
"""

import asyncio
import json
from typing import Dict, List, Tuple
from google_model_client import GoogleModelClient
from llm import IntentDetectionAgent, ConversationState, FunctionSchema, ConversationContext

# Test data sets
CREATE_CLIENT_PROMPTS = [
    "create a new client",
    "I need to add a new client to the system",
    "let's register a new client",
    "can you help me create a client",
    "add client",
    "new client please",
    "I want to create a client profile",
    "set up a new client account",
    "register client",
    "add a new client to our database",
    # Natural speech variations
    "um can we create a new client",
    "uh I'd like to add a client",
    "so... we need to register a new client in the system",
    "create new client profile please",
    "add a client to the database"
]

NAME_INPUTS = [
    "John Smith",
    "ehh, it's david, david rozovsky",
    "um... Sarah Johnson",
    "Robert Chen",
    "Maria Garcia",
    "yeah, the name is Michael O'Brien",
    "it's Jennifer Lee-Wong",
    "uh, William James Anderson III",
    "Dr. Elizabeth Martinez",
    "Thomas van der Berg",
    # Natural speech with filler words and variations
    "ummm... it's... uh... jane, jane thompson",
    "oh yeah, his name is jose, jose martinez gonzalez",
    "so the client's name is... let me think... ah yes, alexander mcallister",
    "it's gonna be... hmm... patricia... patricia anne wilson",
    "the name? oh right, mohammed al rashid",
    "uhhhh mary kate o'sullivan",
    "jean-pierre dubois",
    "you can put down raj, rajesh patel",
    "it's li... li zhang... or maybe zhang li? no, li zhang",
    "anastasia... uh... romanoff... no wait, romanova"
]

PHONE_INPUTS = [
    "212-555-0123",
    "my number is (415) 555-0198",
    "it's 3105550145",
    "call me at 617.555.0167",
    "um, seven one eight five five five zero one five six",
    "yeah, it's 1-800-555-0199",
    "phone is five oh three, five five five, zero one two three",
    "(646) 555-0178",
    "9175550189",
    "the number is eight four seven five five five zero one nine eight",
    # Natural speech variations
    "oh the phone? it's uh... two one two... five five five... zero one seven eight",
    "you can reach them at nine one seven five five five oh one nine nine",
    "the number is... wait let me check... okay it's three zero five five five five zero one four five",
    "ummm eight hundred... no wait... eight six six five five five zero one two three",
    "it's area code four one five then five five five zero one nine eight",
    "phone number? sure, it's six four six dash five five five dash zero one seven eight",
    "you can call at five five five dot zero one two three... oh wait, I forgot the area code, it's two one two",
    "the phone is seven one eight uh five five five and then zero one five six",
    "number is three one zero... no sorry three oh one five five five zero one four five",
    "it's plus one nine one seven five five five zero one eight nine"
]

ADDRESS_INPUTS = [
    "123 Main Street, New York, NY 10001",
    "um, I live at 456 Elm Avenue, Apartment 5B, San Francisco, CA 94102",
    "789 Oak Boulevard, Suite 200, Chicago, IL 60601",
    "yeah it's 321 Pine Street, Boston, MA 02108",
    "1500 Market St, Philadelphia, PA 19102",
    "the address is 2020 K Street NW, Washington, DC 20006",
    "100 Peachtree Street, Atlanta, GA 30303",
    "555 California Street, 12th Floor, Los Angeles, CA 90013",
    "my office is at 1000 Brickell Avenue, Miami, FL 33131",
    "send it to 400 South Hope Street, Unit 1012, Los Angeles, CA 90071",
    # Natural speech variations
    "uh so the address is... let me see... forty two... no four twenty maple avenue in brooklyn new york one one two zero one",
    "they're at... ummm... one fifty... wait, fifteen fifty broadway, apartment three b, in manhattan... zip is one zero zero one nine",
    "oh the address? it's gonna be seven seven seven... uh... seventh avenue... suite twelve hundred... seattle washington nine eight one zero one",
    "so you'll find them at nine hundred north michigan avenue... that's on the... uh... twenty third floor... chicago illinois six zero six one one",
    "the mailing address is... hmm... three three three... no wait three thousand three hundred south las vegas boulevard... las vegas nevada eight nine one zero nine",
    "it's at um twelve hundred pennsylvania avenue northwest... washington dc two zero zero zero four",
    "send everything to five thousand... uh... wilshire boulevard... los angeles... california nine zero zero three six",
    "they're located at... let me think... one market street... suite four thousand... san francisco california nine four one zero five",
    "address is eight eight eight... no sorry... eight zero eight state street... santa barbara... ca... nine three one zero one",
    "you can mail it to... uh... four zero zero... broad street... unit fifteen a... philadelphia p.a. one nine one zero nine"
]

class TestResults:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def add_result(self, test_name: str, passed: bool, details: Dict):
        self.results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self):
        print("\n" + "="*80)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed")
        print("="*80)
        
        if self.failed > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if not result['passed']:
                    print(f"\n- {result['test']}")
                    print(f"  Details: {result['details']}")

async def test_intent_detection(agent: IntentDetectionAgent, prompt: str, test_results: TestResults):
    """Test if the agent correctly detects create_client intent"""
    context = ConversationContext(
        state=ConversationState.DETECTING_INTENT,
        conversation_history=[]
    )
    
    context = await agent.process_user_input(prompt, context)
    
    detected_intent = context.detected_intent
    state = context.state
    
    passed = (detected_intent == 'create_client' and 
              state == ConversationState.GATHERING_PARAMS)
    
    test_results.add_result(
        f"Intent detection: '{prompt[:30]}...'",
        passed,
        {
            'prompt': prompt,
            'detected_intent': detected_intent,
            'state': state.value if state else None
        }
    )
    
    return context if passed else None

async def test_parameter_extraction(agent: IntentDetectionAgent, 
                                  context: ConversationContext,
                                  name: str,
                                  phone: str,
                                  address: str,
                                  test_results: TestResults):
    """Test parameter extraction for name, phone, and address"""
    test_name = f"Parameter extraction: {name[:20]}..."
    
    # Extract name
    context = await agent.process_user_input(name, context)
    
    # Extract phone
    context = await agent.process_user_input(phone, context)
    
    # Extract address
    context = await agent.process_user_input(address, context)
    
    # Get extracted parameters
    extracted_params = context.gathered_parameters or {}
    
    # Check if all parameters were extracted
    has_name = 'client_name' in extracted_params and extracted_params['client_name']
    has_phone = 'phone_number' in extracted_params and extracted_params['phone_number']
    has_address = 'address' in extracted_params and extracted_params['address']
    
    passed = has_name and has_phone and has_address
    
    test_results.add_result(
        test_name,
        passed,
        {
            'input_name': name,
            'input_phone': phone,
            'input_address': address,
            'extracted': extracted_params,
            'missing': {
                'name': not has_name,
                'phone': not has_phone,
                'address': not has_address
            }
        }
    )
    
    return extracted_params if passed else None

async def run_comprehensive_test():
    """Run comprehensive tests on agent client creation"""
    print("Starting comprehensive agent client creation tests...")
    print("="*80)
    
    # Initialize agent
    model_client = GoogleModelClient()
    
    # Create function schema
    create_client_schema = FunctionSchema(
        name="create_client",
        description="Create a new client in the system",
        parameters={
            "client_name": {"type": "string", "description": "Client's full name"},
            "phone_number": {"type": "string", "description": "Client's phone number"},
            "address": {"type": "string", "description": "Client's address"},
            "email": {"type": "string", "description": "Client's email (optional)"}
        },
        required=["client_name", "phone_number", "address"]
    )
    
    agent = IntentDetectionAgent(model_client, [create_client_schema])
    test_results = TestResults()
    
    # Test 1: Intent Detection
    print("\n1. Testing Intent Detection with various prompts...")
    print("-"*60)
    
    successful_contexts = []
    for i, prompt in enumerate(CREATE_CLIENT_PROMPTS):
        context = await test_intent_detection(agent, prompt, test_results)
        if context:
            successful_contexts.append(context)
        print(f"  [{i+1}/{len(CREATE_CLIENT_PROMPTS)}] {'✓' if context else '✗'} {prompt[:50]}...")
    
    # Test 2: Parameter Extraction Combinations
    print("\n2. Testing Parameter Extraction with various inputs...")
    print("-"*60)
    
    # Test all combinations of natural language inputs
    test_combinations = []
    
    # First 10 original combinations
    for i in range(10):
        name_idx = i
        phone_idx = i
        address_idx = i
        test_combinations.append((
            NAME_INPUTS[name_idx],
            PHONE_INPUTS[phone_idx],
            ADDRESS_INPUTS[address_idx]
        ))
    
    # Next 10 with natural language variations
    for i in range(10, min(20, len(NAME_INPUTS))):
        if i < len(NAME_INPUTS) and i < len(PHONE_INPUTS) and i < len(ADDRESS_INPUTS):
            test_combinations.append((
                NAME_INPUTS[i],
                PHONE_INPUTS[i],
                ADDRESS_INPUTS[i]
            ))
    
    for i, (name, phone, address) in enumerate(test_combinations):
        # Use a fresh context for each test
        fresh_context = ConversationContext(
            state=ConversationState.GATHERING_PARAMS,
            detected_intent='create_client',
            gathered_parameters={},
            missing_parameters=['client_name', 'phone_number', 'address'],
            conversation_history=[]
        )
        
        params = await test_parameter_extraction(
            agent, fresh_context, name, phone, address, test_results
        )
        
        if params:
            print(f"  [{i+1}/{len(test_combinations)}] ✓ Successfully extracted all parameters")
            print(f"    Name: {params.get('client_name', 'N/A')}")
            print(f"    Phone: {params.get('phone_number', 'N/A')}")
            print(f"    Address: {params.get('address', 'N/A')[:50]}...")
        else:
            print(f"  [{i+1}/{len(test_combinations)}] ✗ Failed to extract all parameters")
            # Show what was missing
            details = test_results.results[-1]['details']
            missing = details.get('missing', {})
            extracted = details.get('extracted', {})
            if missing.get('name'):
                print(f"    ✗ Name not extracted from: {name[:50]}...")
            else:
                print(f"    ✓ Name: {extracted.get('client_name', 'N/A')}")
            if missing.get('phone'):
                print(f"    ✗ Phone not extracted from: {phone[:50]}...")
            else:
                print(f"    ✓ Phone: {extracted.get('phone_number', 'N/A')}")
            if missing.get('address'):
                print(f"    ✗ Address not extracted from: {address[:50]}...")
            else:
                print(f"    ✓ Address: {extracted.get('address', 'N/A')[:50]}...")
    
    # Test 3: Edge Cases
    print("\n3. Testing Edge Cases...")
    print("-"*60)
    
    edge_cases = [
        ("Test spoken phone", 
         "John Doe",
         "my number is seven one eight five five five zero one five six",
         "123 Test St, NY 10001"),
        ("Test complex name",
         "Dr. María José García-Rodríguez, PhD",
         "212-555-0111",
         "456 Complex Ave, CA 94102"),
        ("Test messy address",
         "Test User",
         "(800) 555-0122",
         "um, yeah, I'm at... uh... 789 Messy Street, apartment B, I think it's Chicago, IL 60601")
    ]
    
    for test_name, name, phone, address in edge_cases:
        fresh_context = ConversationContext(
            state=ConversationState.GATHERING_PARAMS,
            detected_intent='create_client',
            gathered_parameters={},
            missing_parameters=['client_name', 'phone_number', 'address'],
            conversation_history=[]
        )
        
        params = await test_parameter_extraction(
            agent, fresh_context, name, phone, address, test_results
        )
        
        test_results.add_result(
            f"Edge case: {test_name}",
            params is not None,
            {'extracted': params} if params else {'error': 'Failed to extract parameters'}
        )
    
    # Print summary
    test_results.print_summary()
    
    # Show some successful examples
    print("\nSUCCESSFUL EXTRACTION EXAMPLES:")
    print("-"*60)
    success_count = 0
    for result in test_results.results:
        if result['passed'] and 'extracted' in result['details']:
            if success_count < 3:
                print(f"\nTest: {result['test']}")
                print(f"Extracted parameters:")
                params = result['details']['extracted']
                if params:
                    print(f"  - Name: {params.get('client_name', 'N/A')}")
                    print(f"  - Phone: {params.get('phone_number', 'N/A')}")
                    print(f"  - Address: {params.get('address', 'N/A')}")
                success_count += 1

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())