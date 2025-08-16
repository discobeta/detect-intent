from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum

class ConversationState(Enum):
    DETECTING_INTENT = "detecting_intent"
    GATHERING_PARAMS = "gathering_params"
    READY_TO_EXECUTE = "ready_to_execute"
    CLARIFICATION_NEEDED = "clarification_needed"

@dataclass
class FunctionSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

@dataclass
class ConversationContext:
    state: ConversationState
    detected_intent: Optional[str] = None
    gathered_parameters: Dict[str, Any] = None
    missing_parameters: List[str] = None
    conversation_history: List[Dict[str, str]] = None
    confidence: float = 0.0

class IntentDetectionAgent:

    def __init__(self, model_client, function_schemas: List[FunctionSchema]):
        self.model_client = model_client
        self.function_schemas = {fs.name: fs for fs in function_schemas}
        self.schema_text = self._build_schema_text()
    
    def _build_schema_text(self) -> str:
        """Convert function schemas to text for the LLM"""
        schema_parts = []
        for name, schema in self.function_schemas.items():
            schema_parts.append(f"""
Function: {name}
Description: {schema.description}
Parameters: {json.dumps(schema.parameters, indent=2)}
Required: {schema.required}
""")
        return "\n".join(schema_parts)
    
    async def process_user_input(self, user_input: str, context: ConversationContext) -> ConversationContext:
        """Main processing function"""
        
        if context.state == ConversationState.DETECTING_INTENT:
            return await self._detect_intent(user_input, context)
        
        elif context.state == ConversationState.GATHERING_PARAMS:
            return await self._gather_parameters(user_input, context)
        
        elif context.state == ConversationState.CLARIFICATION_NEEDED:
            return await self._handle_clarification(user_input, context)
        
        return context
    
    async def _detect_intent(self, user_input: str, context: ConversationContext) -> ConversationContext:
        """Detect user intent from input"""
        
        prompt = f"""
Available functions:
{self.schema_text}

User input: "{user_input}"

Analyze the user input and determine:
1. Which function they want to use (or "unknown" if unclear)
2. What parameters you can extract from their input
3. Confidence level (0-1)

Respond in JSON format:
{{
    "intent": "function_name_or_unknown",
    "extracted_parameters": {{}},
    "confidence": 0.8,
    "reasoning": "explanation"
}}
"""
        
        response = await self.model_client.generate(prompt)
        result = json.loads(response)
        
        if result["intent"] != "unknown" and result["confidence"] > 0.7:
            context.detected_intent = result["intent"]
            context.gathered_parameters = result.get("extracted_parameters", {})
            context.state = ConversationState.GATHERING_PARAMS
            context.confidence = result["confidence"]
            
            # Check if we have all required parameters
            missing = self._get_missing_parameters(context)
            if not missing:
                context.state = ConversationState.READY_TO_EXECUTE
            else:
                context.missing_parameters = missing
        else:
            context.state = ConversationState.CLARIFICATION_NEEDED
        
        return context
    
    async def _gather_parameters(self, user_input: str, context: ConversationContext) -> ConversationContext:
        """Extract parameters from user response"""
        
        schema = self.function_schemas[context.detected_intent]
        
        prompt = f"""
Function: {schema.name}
Parameters needed: {json.dumps(schema.parameters, indent=2)}
Required parameters: {schema.required}
Currently gathered: {json.dumps(context.gathered_parameters, indent=2)}
Missing parameters: {context.missing_parameters}

User response: "{user_input}"

Extract any parameter values from the user response and update the gathered parameters.

Respond in JSON format:
{{
    "updated_parameters": {{}},
    "still_missing": [],
    "ready_to_execute": false
}}
"""
        
        response = await self.model_client.generate(prompt)
        result = json.loads(response)
        
        # Update gathered parameters
        context.gathered_parameters.update(result["updated_parameters"])
        context.missing_parameters = result["still_missing"]
        
        if result["ready_to_execute"] or not context.missing_parameters:
            context.state = ConversationState.READY_TO_EXECUTE
        
        return context
    
    async def _handle_clarification(self, user_input: str, context: ConversationContext) -> ConversationContext:
        """Handle cases where intent is unclear"""
        
        prompt = f"""
Available functions:
{self.schema_text}

User input: "{user_input}"
Previous conversation: {context.conversation_history[-3:] if context.conversation_history else []}

The user's intent was unclear. Try again to determine what they want to do.

Respond in JSON format:
{{
    "intent": "function_name_or_ask_for_clarification",
    "clarification_question": "What would you like me to help you with?",
    "confidence": 0.5
}}
"""
        
        response = await self.model_client.generate(prompt)
        result = json.loads(response)
        
        if result["intent"] != "ask_for_clarification":
            context.detected_intent = result["intent"]
            context.state = ConversationState.GATHERING_PARAMS
        
        return context
    
    def _get_missing_parameters(self, context: ConversationContext) -> List[str]:
        """Check which required parameters are still missing"""
        if not context.detected_intent:
            return []
        
        schema = self.function_schemas[context.detected_intent]
        missing = []
        
        for param in schema.required:
            if param not in context.gathered_parameters or context.gathered_parameters[param] is None:
                missing.append(param)
        
        return missing
    
    async def generate_response(self, context: ConversationContext) -> str:
        """Generate appropriate response based on current state"""
        
        if context.state == ConversationState.READY_TO_EXECUTE:
            return f"""
Ready to execute: {context.detected_intent}
Parameters: {json.dumps(context.gathered_parameters, indent=2)}
Confidence: {context.confidence}
"""
        
        elif context.state == ConversationState.GATHERING_PARAMS:
            return await self._generate_parameter_question(context)
        
        elif context.state == ConversationState.CLARIFICATION_NEEDED:
            return "I'm not sure what you'd like me to help you with. Could you please clarify what function you want to use?"
        
        return "I'm processing your request..."
    
    async def _generate_parameter_question(self, context: ConversationContext) -> str:
        """Generate a natural question to gather missing parameters"""
        
        schema = self.function_schemas[context.detected_intent]
        missing = context.missing_parameters
        
        prompt = f"""
Function: {schema.name} - {schema.description}
Missing parameters: {missing}
Parameter definitions: {json.dumps({p: schema.parameters.get(p, {}) for p in missing}, indent=2) if missing else "No missing parameters"}

Generate a natural, conversational question to ask the user for these missing parameters.
Keep it friendly and specific. Ask for one or two parameters at a time if there are many.

Just return the question text, nothing else.
"""
        
        return await self.model_client.generate(prompt)
