#!/usr/bin/env python3
"""
Test script for M2 MCP Server
Validates that all tools work correctly
"""

import asyncio
import json
from mcp_server import (
    load_model_and_data,
    predict_m2_future,
    get_m2_current,
    get_m2_statistics
)


async def test_tools():
    """Test all MCP server tools"""
    
    print("=" * 60)
    print("Testing M2 MCP Server Tools")
    print("=" * 60)
    
    # Load model and data
    print("\n1. Loading model and data...")
    try:
        load_model_and_data()
        print("   ✓ Model and data loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model/data: {e}")
        return
    
    # Test predict_m2_future
    print("\n2. Testing predict_m2_future tool...")
    try:
        result = await predict_m2_future({"months": 6})
        data = json.loads(result[0].text)
        print(f"   ✓ Generated {len(data['predictions'])} predictions")
        print(f"   First prediction: {data['predictions'][0]['date']} = ${data['predictions'][0]['predicted_m2_billions']}B")
        print(f"   Last prediction: {data['predictions'][-1]['date']} = ${data['predictions'][-1]['predicted_m2_billions']}B")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test get_m2_current
    print("\n3. Testing get_m2_current tool...")
    try:
        result = await get_m2_current({})
        data = json.loads(result[0].text)
        print(f"   ✓ Current M2: ${data['m2_billions']}B as of {data['observation_date']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test get_m2_statistics
    print("\n4. Testing get_m2_statistics tool...")
    for period in ["1year", "5year", "all"]:
        try:
            result = await get_m2_statistics({"period": period})
            data = json.loads(result[0].text)
            print(f"   ✓ {period}: Mean=${data['statistics']['mean_m2_billions']}B, "
                  f"Growth={data['growth']['annualized_growth_percent']}% annualized")
        except Exception as e:
            print(f"   ✗ Error for {period}: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_tools())