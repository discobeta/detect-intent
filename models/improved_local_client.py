#!/usr/bin/env python3
"""
Wrapper for backward compatibility - redirects to google_model_client
"""

from .google_model_client import GoogleModelClient

# Alias for backward compatibility
ImprovedLocalClient = GoogleModelClient

def create_improved_local_client(model_type: str = "optimized", **kwargs):
    """Create a Google model client"""
    return GoogleModelClient()