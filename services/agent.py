"""
Agent service for handling conversational interactions
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

from .conversation_manager import IntentDetectionAgent, ConversationContext, ConversationState, FunctionSchema
from models.google_model_client import GoogleModelClient



class AgentService:
    """Service for managing conversational agent interactions"""
    
    def __init__(self, config_path: str = "config/agent_config.json"):
        """Initialize the agent service with configuration"""
        self.config = self._load_config(config_path)
        self.client = GoogleModelClient(model_name=self.config["model"]["name"])
        self.agent = self._initialize_agent()
        self.context = self._reset_context()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_file = Path(__file__).parent.parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def _initialize_agent(self) -> IntentDetectionAgent:
        """Initialize the agent with function schemas from config"""
        schemas = []
        for func_config in self.config["functions"]:
            schema = FunctionSchema(
                name=func_config["name"],
                description=func_config["description"],
                parameters=func_config["parameters"],
                required=func_config["required"]
            )
            schemas.append(schema)
            
        return IntentDetectionAgent(
            model_client=self.client,
            function_schemas=schemas
        )
    
    def _reset_context(self) -> ConversationContext:
        """Reset the conversation context"""
        return ConversationContext(
            state=ConversationState.DETECTING_INTENT,
            gathered_parameters={},
            conversation_history=[]
        )
    
    async def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message and return the response
        
        Returns:
            Dict containing:
            - response: The agent's response text
            - state: Current conversation state
            - intent: Detected intent (if any)
            - parameters: Gathered parameters
            - ready_to_execute: Whether function is ready to execute
        """
        # Process user input
        self.context = await self.agent.process_user_input(user_input, self.context)
        
        # Generate response
        response = await self.agent.generate_response(self.context)
        
        # Check if ready to execute
        ready_to_execute = self.context.state == ConversationState.READY_TO_EXECUTE
        
        # Prepare result
        result = {
            "response": response,
            "state": self.context.state.value,
            "intent": self.context.detected_intent,
            "parameters": self.context.gathered_parameters,
            "ready_to_execute": ready_to_execute
        }
        
        # Reset context if function is ready
        if ready_to_execute:
            self.context = self._reset_context()
            
        return result
    
    async def start_interactive_session(self):
        """Start an interactive conversation session"""
        print(f"\n🤖 {self.config['agent']['name']}")
        print("=" * 60)
        print(f"{self.config['agent']['description']}")
        print(f"\nAvailable functions:")
        
        for func in self.config['functions']:
            print(f"  - {func['name']}: {func['description']}")
        
        print(f"\n{self.config['agent']['instructions']}")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in self.config['agent']['exit_commands']:
                    print(f"\n{self.config['agent']['goodbye_message']}")
                    break
                
                # Process the message
                result = await self.process_message(user_input)
                
                # Display response
                print(f"\n{self.config['agent']['name']}: {result['response']}")
                
                # Show debug info if enabled
                if self.config.get('debug', False):
                    if result['intent']:
                        print(f"[Intent: {result['intent']}]")
                    if result['parameters']:
                        print(f"[Parameters: {json.dumps(result['parameters'], indent=2)}]")
                
                # Show execution message if ready
                if result['ready_to_execute']:
                    print(f"\n✅ Ready to execute '{result['intent']}' with:")
                    for key, value in result['parameters'].items():
                        print(f"   {key}: {value}")
                    print("\n" + "-" * 60)
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.config['agent']['goodbye_message']}")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                if self.config.get('debug', False):
                    import traceback
                    traceback.print_exc()


def create_agent_service(config_path: Optional[str] = None) -> AgentService:
    """Factory function to create an agent service"""
    if config_path is None:
        config_path = "config/agent_config.json"
    
    # Set environment variable for macOS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    return AgentService(config_path)