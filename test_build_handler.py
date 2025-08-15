#!/usr/bin/env python3
"""
Quick test script to verify the build_handler function works with revision parameter
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'berkeley-function-call-leaderboard'))

from bfcl_eval._llm_response_generation import build_handler

def test_build_handler():
    """Test that build_handler works with handlers that don't support revision"""
    
    # Test with a handler that doesn't support revision (QwenFCHandler)
    try:
        handler = build_handler("Qwen/Qwen3-4B-Instruct-2507-FC", 0.001, revision="main")
        print("✓ Successfully created QwenFCHandler with revision parameter")
        print(f"✓ Handler type: {type(handler)}")
        print(f"✓ Handler has is_fc_model: {hasattr(handler, 'is_fc_model')}")
    except Exception as e:
        print(f"✗ Failed to create QwenFCHandler: {e}")
        return False
    
    # Test with a handler that supports revision (SmolLM3Handler)
    try:
        handler2 = build_handler("HuggingFaceTB/SmolLM3-SFT", 0.001, revision="v27.00-step-000000172")
        print("✓ Successfully created SmolLM3Handler with revision parameter")
        print(f"✓ Handler type: {type(handler2)}")
        print(f"✓ Handler revision: {getattr(handler2, 'revision', 'Not found')}")
    except Exception as e:
        print(f"✗ Failed to create SmolLM3Handler: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_build_handler()
    sys.exit(0 if success else 1)