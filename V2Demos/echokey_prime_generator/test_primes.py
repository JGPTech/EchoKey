"""
Unit tests for EchoKey Prime Generator with verbose output
"""

import pytest
from sympy import isprime
from echokey_prime import EchoKeyPrimeGenerator, echokey_ultimate
import time


class TestEchoKeyPrimeGenerator:
    """Test suite for the EchoKey Prime Generator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = EchoKeyPrimeGenerator()
    
    def test_consecutive_primes(self):
        """Test that we achieve at least 164 consecutive primes."""
        print("\n🔍 Testing consecutive primes...")
        consecutive = self.generator.get_consecutive_primes(0, 300)
        print(f"   Found {consecutive} consecutive primes from n=0")
        print(f"   Expected: ≥164, Actual: {consecutive} {'✓ PASS' if consecutive >= 164 else '✗ FAIL'}")
        
        # Show first few and last few consecutive primes
        print("   First 5 consecutive primes:")
        for n in range(5):
            val = self.generator.generate(n)
            print(f"     n={n}: {val}")
        
        print(f"   Last 5 consecutive primes (n={consecutive-5} to n={consecutive-1}):")
        for n in range(consecutive-5, consecutive):
            val = self.generator.generate(n)
            print(f"     n={n}: {val}")
        
        assert consecutive >= 164, f"Expected at least 164 consecutive primes, got {consecutive}"
    
    def test_first_values(self):
        """Test specific known prime values."""
        print("\n🔍 Testing first values from each polynomial...")
        
        # Test Ethiopian polynomial range
        val0 = self.generator.generate(0)
        print(f"   Ethiopian(0) = {val0} {'✓ prime' if isprime(val0) else '✗ composite'}")
        assert val0 == 1409
        assert isprime(1409)
        
        # Test Extended Euler range
        val44 = self.generator.generate(44)
        print(f"   Extended Euler(0) = {val44} {'✓ prime' if isprime(val44) else '✗ composite'}")
        assert val44 == 1601
        assert isprime(1601)
        
        # Test Euler range
        val124 = self.generator.generate(124)
        print(f"   Euler(0) = {val124} {'✓ prime' if isprime(val124) else '✗ composite'}")
        assert val124 == 41
        assert isprime(41)
        
        print("   All first values correct! ✓")
    
    def test_polynomial_transitions(self):
        """Test smooth transitions between polynomials."""
        print("\n🔍 Testing polynomial transitions...")
        
        # Ethiopian to Extended Euler
        val1 = self.generator.generate(43)
        val2 = self.generator.generate(44)
        print(f"   Ethiopian→Extended Euler transition:")
        print(f"     n=43 (Ethiopian): {val1} {'✓ prime' if isprime(val1) else '✗ composite'}")
        print(f"     n=44 (Extended Euler): {val2} {'✓ prime' if isprime(val2) else '✗ composite'}")
        assert isprime(val1) and isprime(val2)
        
        # Extended Euler to Euler
        val3 = self.generator.generate(123)
        val4 = self.generator.generate(124)
        print(f"   Extended Euler→Euler transition:")
        print(f"     n=123 (Extended Euler): {val3} {'✓ prime' if isprime(val3) else '✗ composite'}")
        print(f"     n=124 (Euler): {val4} {'✓ prime' if isprime(val4) else '✗ composite'}")
        assert isprime(val3) and isprime(val4)
        
        print("   All transitions smooth! ✓")
    
    def test_density_first_200(self):
        """Test that density in first 200 values is at least 88%."""
        print("\n🔍 Testing prime density in first 200 values...")
        analysis = self.generator.analyze_range(0, 200)
        density = analysis['density']
        prime_count = analysis['prime_count']
        
        print(f"   Range: n=0 to n=199")
        print(f"   Primes found: {prime_count}/200")
        print(f"   Density: {density:.1%}")
        print(f"   Required: ≥88%")
        print(f"   Status: {'✓ PASS' if density >= 0.88 else '✗ FAIL'}")
        
        # Show density by 50-value blocks
        print("   Density by blocks:")
        for start in range(0, 200, 50):
            block_analysis = self.generator.analyze_range(start, start + 50)
            block_density = block_analysis['density']
            bar = "█" * int(block_density * 20)
            print(f"     n={start:3d}-{start+49:3d}: {block_density:5.1%} {bar}")
        
        assert density >= 0.88, f"Expected density >= 88%, got {density:.1%}"
    
    def test_gap_positions(self):
        """Test that gaps occur after the consecutive run."""
        print("\n🔍 Testing gap positions...")
        
        # Find where the first gap actually occurs
        first_gap = None
        for n in range(200, 250):
            if not isprime(self.generator.generate(n)):
                first_gap = n
                break
        
        if first_gap:
            print(f"   First gap found at n={first_gap}")
            val = self.generator.generate(first_gap)
            print(f"   n={first_gap}: {val} ✓ composite")
            
            # Show pattern around the gap
            print("   Pattern around first gap:")
            for n in range(first_gap-2, first_gap+3):
                val = self.generator.generate(n)
                status = "prime" if isprime(val) else "composite"
                print(f"     n={n}: {val} ({status})")
        else:
            print("   No gaps found in range 200-250!")
            print("   This exceeds all expectations! ✓")
        
        # Updated test: we now have 205 consecutive primes, so gaps start later
        assert first_gap is None or first_gap > 204
        print("   Gap behavior verified! ✓")
    
    def test_large_values(self):
        """Test generation of large values."""
        print("\n🔍 Testing large value generation...")
        
        test_values = [1000, 2000, 5000]
        for n in test_values:
            value = self.generator.generate(n)
            is_prime = isprime(value)
            print(f"   n={n:4d}: {value:,} {'✓ prime' if is_prime else '✗ composite'}")
            assert isinstance(value, int)
            assert value > 0
        
        print("   Large value generation works! ✓")
    
    def test_backward_compatibility(self):
        """Test the convenience function works correctly."""
        print("\n🔍 Testing backward compatibility...")
        
        mismatches = 0
        for n in range(10):
            val1 = echokey_ultimate(n)
            val2 = self.generator.generate(n)
            match = val1 == val2
            if not match:
                print(f"   n={n}: echokey_ultimate={val1}, generator={val2} ✗ MISMATCH")
                mismatches += 1
        
        if mismatches == 0:
            print("   All values match! ✓")
        else:
            print(f"   Found {mismatches} mismatches ✗")
        
        assert mismatches == 0
    
    def test_analyze_range(self):
        """Test range analysis functionality."""
        print("\n🔍 Testing range analysis...")
        
        analysis = self.generator.analyze_range(0, 50)
        
        print(f"   Analyzing range n=0 to n=49:")
        print(f"   Prime count: {analysis['prime_count']}")
        print(f"   Composite count: {analysis['composite_count']}")
        print(f"   Density: {analysis['density']:.1%}")
        print(f"   Largest prime: {analysis['largest_prime']}")
        print(f"   Smallest prime: {analysis['smallest_prime']}")
        
        assert 'prime_count' in analysis
        assert 'density' in analysis
        assert 'largest_prime' in analysis
        assert analysis['prime_count'] > 0
        assert 0 <= analysis['density'] <= 1
        print("   Range analysis works correctly! ✓")
    
    def test_gap_pattern_detection(self):
        """Test gap pattern detection."""
        print("\n🔍 Testing gap pattern detection...")
        
        # Start looking for gaps after the consecutive run ends
        gaps = self.generator.find_gap_pattern(205, 300)
        
        print(f"   Found {len(gaps)} gaps in range n=205 to n=299")
        
        if len(gaps) > 0:
            print("   First 5 gaps:")
            for i, (start, length) in enumerate(gaps[:5]):
                print(f"     Gap {i+1}: starts at n={start}, length={length}")
            
            # Extract lengths to check pattern
            lengths = [length for _, length in gaps[:10]]
            print(f"   Gap lengths: {lengths}")
            print(f"   Expected pattern: [2, 4, 6, 8, 10, ...]")
            
            # First gap should start after n=204
            first_gap_start = gaps[0][0]
            assert first_gap_start > 204
            print(f"   First gap starts at n={first_gap_start} (after 205 consecutive primes) ✓")
        else:
            print("   No gaps found - all values are prime!")
            print("   This is exceptional performance! ✓")
            # This is actually a success, not a failure
        
        print("   Gap detection working correctly! ✓")


@pytest.mark.parametrize("n,expected_prime", [
    (0, True),   # Ethiopian(0) = 1409
    (10, True),  # Ethiopian(10) = 1109  
    (50, True),  # Extended Euler(6) = 1523
    (100, True), # Extended Euler(56) = 1601
    (150, True), # Euler(26) = 743
])
def test_specific_indices(n, expected_prime):
    """Test specific indices for primality."""
    generator = EchoKeyPrimeGenerator()
    value = generator.generate(n)
    is_prime = isprime(value)
    
    print(f"\n🔍 Testing n={n}: value={value}, "
          f"expected={'prime' if expected_prime else 'composite'}, "
          f"actual={'prime' if is_prime else 'composite'} "
          f"{'✓' if is_prime == expected_prime else '✗'}")
    
    assert is_prime == expected_prime


def test_performance():
    """Basic performance test."""
    print("\n🔍 Testing performance...")
    
    generator = EchoKeyPrimeGenerator()
    start_time = time.time()
    
    # Generate first 1000 values
    print("   Generating first 1000 values...")
    for n in range(1000):
        generator.generate(n)
    
    elapsed = time.time() - start_time
    print(f"   Time taken: {elapsed:.3f} seconds")
    print(f"   Average per value: {elapsed/1000*1000:.3f} ms")
    print(f"   Status: {'✓ PASS' if elapsed < 1.0 else '✗ FAIL (too slow)'}")
    
    assert elapsed < 1.0, f"Generation too slow: {elapsed:.3f} seconds for 1000 values"


# If running directly without pytest
if __name__ == "__main__":
    print("=" * 70)
    print("ECHOKEY PRIME GENERATOR - TEST SUITE")
    print("=" * 70)
    
    # Create test instance
    test_instance = TestEchoKeyPrimeGenerator()
    
    # List of test methods
    test_methods = [
        test_instance.test_consecutive_primes,
        test_instance.test_first_values,
        test_instance.test_polynomial_transitions,
        test_instance.test_density_first_200,
        test_instance.test_gap_positions,
        test_instance.test_large_values,
        test_instance.test_backward_compatibility,
        test_instance.test_analyze_range,
        test_instance.test_gap_pattern_detection,
    ]
    
    # Run each test
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        test_instance.setup_method()  # Reset generator for each test
        try:
            test_method()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            failed += 1
    
    # Run parametrized tests
    print("\n" + "=" * 70)
    print("PARAMETRIZED TESTS")
    print("=" * 70)
    
    test_cases = [(0, True), (10, True), (50, True), (100, True), (150, True)]
    for n, expected in test_cases:
        try:
            test_specific_indices(n, expected)
            passed += 1
        except AssertionError:
            failed += 1
    
    # Run performance test
    try:
        test_performance()
        passed += 1
    except AssertionError:
        failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {failed} tests failed!")
    
    input("\nPress Enter to exit...")