# Conversational Agent with Intent Detection

A simple conversational agent using Google's Flan-T5 model for intent detection and parameter gathering.

## Features

- Natural language intent detection
- Parameter extraction from conversational responses
- Advanced name extraction using spaCy NER
- Address extraction and validation using multiple methods (pyap, usaddress)
- Phone number extraction and validation using phonenumbers library
- Configurable functions and parameters
- Clean service-based architecture

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Run the agent:
```bash
python run.py
```

## Configuration

Edit `config/agent_config.json` to:
- Add or modify functions
- Change agent personality
- Enable debug mode
- Customize messages

## Project Structure

```
tts/
├── run.py                  # Main entry point
├── config/
│   └── agent_config.json   # Agent configuration
├── services/
│   ├── agent.py           # Agent service logic
│   ├── name.py            # Name extraction service
│   ├── address.py         # Address extraction service
│   └── phone.py           # Phone number extraction service
├── google_model_client.py  # Google Flan-T5 model client
├── llm.py                 # Core LLM integration
└── improved_local_client.py # Compatibility wrapper
```

## Usage Example

```
You: create a new client
Agent: What is the client's full name?
You: ehh, it's david, david rozovsky
Agent: What is the client's phone number?
You: 212-555-0123
Agent: What is the client's address?
You: 123 Main St, New York, NY 10001
Agent: Ready to execute 'create_client'
```