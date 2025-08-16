#!/usr/bin/env python3
"""
Simple interactive agent runner
"""

import asyncio
from services.agent import create_agent_service


async def main():
    """Run the interactive agent"""
    try:
        # Create and start the agent service
        agent = create_agent_service()
        await agent.start_interactive_session()
        
    except FileNotFoundError as e:
        print(f"\n❌ Configuration error: {e}")
        print("Please ensure config/agent_config.json exists")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())