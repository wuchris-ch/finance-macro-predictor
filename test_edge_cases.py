#!/usr/bin/env python3
"""
Edge case and integration tests for M2 MCP Server
Tests boundary conditions, error handling, and invalid inputs
"""

import asyncio
import json
import time
from mcp_server import (
    load_model_and_data,
    predict_m2_future,
    get_m2_current,
    get_m2_statistics
)


async def test_edge_cases():
    """Test edge cases and error handling"""
    
    print("=" * 60)
    print("Edge Case and Integration Testing")
    print("=" * 60)
    
    # Load model and data first
    print("\n[SETUP] Loading model and data...")
    try:
        load_model_and_data()
        print("✓ Model and data loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model/data: {e}")
        return
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    # Test 1: Minimum months (1)
    print("Test 1: Predict with minimum months (1)")
    try:
        start_time = time.time()
        result = await predict_m2_future({"months": 1})
        elapsed = time.time() - start_time
        data = json.loads(result[0].text)
        assert len(data['predictions']) == 1, "Should return exactly 1 prediction"
        print(f"✓ PASS - Generated 1 prediction in {elapsed:.3f}s")
        print(f"  Prediction: {data['predictions'][0]['date']} = ${data['predictions'][0]['predicted_m2_billions']}B")
        test_results["passed"] += 1
    except Exception as e:
        print(f"✗ FAIL - {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Test 1: {e}")
    
    # Test 2: Maximum months (24)
    print("\nTest 2: Predict with maximum months (24)")
    try:
        start_time = time.time()
        result = await predict_m2_future({"months": 24})
        elapsed = time.time() - start_time
        data = json.loads(result[0].text)
        assert len(data['predictions']) == 24, "Should return exactly 24 predictions"
        print(f"✓ PASS - Generated 24 predictions in {elapsed:.3f}s")
        print(f"  First: {data['predictions'][0]['date']} = ${data['predictions'][0]['predicted_m2_billions']}B")
        print(f"  Last: {data['predictions'][-1]['date']} = ${data['predictions'][-1]['predicted_m2_billions']}B")
        test_results["passed"] += 1
    except Exception as e:
        print(f"✗ FAIL - {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Test 2: {e}")
    
    # Test 3: Invalid months (0)
    print("\nTest 3: Predict with invalid months (0)")
    try:
        result = await predict_m2_future({"months": 0})
        data = json.loads(result[0].text)
        if "error" in data:
            print(f"✓ PASS - Correctly rejected with error: {data['error']}")
            test_results["passed"] += 1
        else:
            print(f"✗ FAIL - Should have returned error for months=0")
            test_results["failed"] += 1
            test_results["errors"].append("Test 3: No error for months=0")
    except Exception as e:
        print(f"✓ PASS - Correctly raised exception: {e}")
        test_results["passed"] += 1
    
    # Test 4: Invalid months (100)
    print("\nTest 4: Predict with invalid months (100)")
    try:
        result = await predict_m2_future({"months": 100})
        data = json.loads(result[0].text)
        if "error" in data:
            print(f"✓ PASS - Correctly rejected with error: {data['error']}")
            test_results["passed"] += 1
        else:
            print(f"✗ FAIL - Should have returned error for months=100")
            test_results["failed"] += 1
            test_results["errors"].append("Test 4: No error for months=100")
    except Exception as e:
        print(f"✓ PASS - Correctly raised exception: {e}")
        test_results["passed"] += 1
    
    # Test 5: Invalid months (negative)
    print("\nTest 5: Predict with negative months (-5)")
    try:
        result = await predict_m2_future({"months": -5})
        data = json.loads(result[0].text)
        if "error" in data:
            print(f"✓ PASS - Correctly rejected with error: {data['error']}")
            test_results["passed"] += 1
        else:
            print(f"✗ FAIL - Should have returned error for months=-5")
            test_results["failed"] += 1
            test_results["errors"].append("Test 5: No error for months=-5")
    except Exception as e:
        print(f"✓ PASS - Correctly raised exception: {e}")
        test_results["passed"] += 1
    
    # Test 6: Invalid period for statistics
    print("\nTest 6: Statistics with invalid period")
    try:
        result = await get_m2_statistics({"period": "invalid_period"})
        data = json.loads(result[0].text)
        if "error" in data:
            print(f"✓ PASS - Correctly rejected with error: {data['error']}")
            test_results["passed"] += 1
        else:
            print(f"✗ FAIL - Should have returned error for invalid period")
            test_results["failed"] += 1
            test_results["errors"].append("Test 6: No error for invalid period")
    except Exception as e:
        print(f"✓ PASS - Correctly raised exception: {e}")
        test_results["passed"] += 1
    
    # Test 7: Current M2 with empty parameters
    print("\nTest 7: Get current M2 (no parameters)")
    try:
        start_time = time.time()
        result = await get_m2_current({})
        elapsed = time.time() - start_time
        data = json.loads(result[0].text)
        assert "m2_billions" in data, "Should return m2_billions"
        assert "observation_date" in data, "Should return observation_date"
        print(f"✓ PASS - Retrieved current M2 in {elapsed:.3f}s")
        print(f"  Current M2: ${data['m2_billions']}B as of {data['observation_date']}")
        test_results["passed"] += 1
    except Exception as e:
        print(f"✗ FAIL - {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Test 7: {e}")
    
    # Test 8: All valid periods for statistics
    print("\nTest 8: Statistics for all valid periods")
    valid_periods = ["1year", "5year", "all"]
    for period in valid_periods:
        try:
            start_time = time.time()
            result = await get_m2_statistics({"period": period})
            elapsed = time.time() - start_time
            data = json.loads(result[0].text)
            assert "statistics" in data, f"Should return statistics for {period}"
            assert "growth" in data, f"Should return growth for {period}"
            print(f"✓ PASS - {period}: Retrieved in {elapsed:.3f}s")
            print(f"  Mean: ${data['statistics']['mean_m2_billions']}B, Growth: {data['growth']['annualized_growth_percent']}%")
            test_results["passed"] += 1
        except Exception as e:
            print(f"✗ FAIL - {period}: {e}")
            test_results["failed"] += 1
            test_results["errors"].append(f"Test 8 ({period}): {e}")
    
    # Test 9: Prediction consistency (run same prediction twice)
    print("\nTest 9: Prediction consistency check")
    try:
        result1 = await predict_m2_future({"months": 3})
        result2 = await predict_m2_future({"months": 3})
        data1 = json.loads(result1[0].text)
        data2 = json.loads(result2[0].text)
        
        # Check if predictions are identical
        pred1 = data1['predictions']
        pred2 = data2['predictions']
        
        if pred1 == pred2:
            print(f"✓ PASS - Predictions are consistent across multiple calls")
            test_results["passed"] += 1
        else:
            print(f"✗ FAIL - Predictions differ between calls (may indicate randomness)")
            test_results["failed"] += 1
            test_results["errors"].append("Test 9: Inconsistent predictions")
    except Exception as e:
        print(f"✗ FAIL - {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Test 9: {e}")
    
    # Test 10: Performance test - multiple rapid predictions
    print("\nTest 10: Performance test (10 rapid predictions)")
    try:
        start_time = time.time()
        for i in range(10):
            await predict_m2_future({"months": 6})
        elapsed = time.time() - start_time
        avg_time = elapsed / 10
        print(f"✓ PASS - Completed 10 predictions in {elapsed:.3f}s (avg: {avg_time:.3f}s)")
        test_results["passed"] += 1
    except Exception as e:
        print(f"✗ FAIL - {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Test 10: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['passed'] + test_results['failed']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    if test_results["errors"]:
        print("\nErrors encountered:")
        for error in test_results["errors"]:
            print(f"  - {error}")
    
    print("=" * 60)
    
    return test_results


if __name__ == "__main__":
    results = asyncio.run(test_edge_cases())
    exit(0 if results["failed"] == 0 else 1)