"""
Basic usage examples for EchoKey Prime Generator
"""

from echokey_prime import EchoKeyPrimeGenerator, echokey_ultimate
from sympy import isprime


def example_1_basic_generation():
    """Example 1: Generate and check individual primes."""
    print("Example 1: Basic Prime Generation")
    print("-" * 40)
    
    # Using the convenience function
    for n in range(10):
        value = echokey_ultimate(n)
        is_prime = isprime(value)
        print(f"n={n}: {value} {'(prime)' if is_prime else '(composite)'}")
    print()


def example_2_consecutive_primes():
    """Example 2: Find consecutive primes."""
    print("Example 2: Consecutive Primes")
    print("-" * 40)
    
    generator = EchoKeyPrimeGenerator()
    consecutive = generator.get_consecutive_primes(0, 300)
    
    print(f"Found {consecutive} consecutive primes starting from n=0")
    print(f"This {'exceeds' if consecutive > 164 else 'meets'} our expected 164!")
    print()


def example_3_density_analysis():
    """Example 3: Analyze prime density in different ranges."""
    print("Example 3: Prime Density Analysis")
    print("-" * 40)
    
    generator = EchoKeyPrimeGenerator()
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
    
    for start, end in ranges:
        analysis = generator.analyze_range(start, end)
        print(f"Range n={start}-{end-1}:")
        print(f"  Primes: {analysis['prime_count']}/{end-start}")
        print(f"  Density: {analysis['density']:.1%}")
        if analysis['largest_prime']:
            print(f"  Largest: {analysis['largest_prime']}")
    print()


def example_4_gap_patterns():
    """Example 4: Analyze gap patterns."""
    print("Example 4: Gap Pattern Analysis")
    print("-" * 40)
    
    generator = EchoKeyPrimeGenerator()
    gaps = generator.find_gap_pattern(160, 220)
    
    print("Gaps found (position, length):")
    for gap_start, gap_length in gaps[:5]:  # Show first 5 gaps
        print(f"  Gap at n={gap_start}, length={gap_length}")
    
    # Extract just the lengths
    lengths = [length for _, length in gaps[:10]]
    print(f"\nGap lengths: {lengths}")
    print("Expected arithmetic progression: [2, 4, 6, 8, 10, ...]")
    print()


def example_5_large_primes():
    """Example 5: Generate large primes."""
    print("Example 5: Large Prime Generation")
    print("-" * 40)
    
    generator = EchoKeyPrimeGenerator()
    indices = [500, 1000, 2000, 5000, 10000]
    
    results = generator.generate_large_primes(indices)
    
    for n, value, is_prime in results:
        status = "✓ prime" if is_prime else "✗ composite"
        print(f"n={n:5d}: {value:,} ({status})")
    print()


def example_6_polynomial_ranges():
    """Example 6: Show which polynomial is used for different ranges."""
    print("Example 6: Polynomial Usage Map")
    print("-" * 40)
    
    generator = EchoKeyPrimeGenerator()
    
    # Sample from each polynomial's range
    samples = [
        (0, "Ethiopian"),
        (20, "Ethiopian"),
        (43, "Ethiopian"),
        (44, "Extended Euler"),
        (80, "Extended Euler"),
        (123, "Extended Euler"),
        (124, "Euler"),
        (150, "Euler"),
        (200, "Euler (extended)"),
        (500, "Euler (extended)")
    ]
    
    for n, expected_poly in samples:
        value = generator.generate(n)
        print(f"n={n:3d}: {value:6d} (using {expected_poly})")
    print()


def example_7_verification():
    """Example 7: Verify the 205 consecutive primes claim."""
    print("Example 7: Verification of 205 Consecutive Primes")
    print("-" * 50)
    
    generator = EchoKeyPrimeGenerator()
    
    # Check each value from 0 to 205
    all_prime = True
    for n in range(206):
        value = generator.generate(n)
        if not isprime(value):
            print(f"First composite at n={n}: {value}")
            all_prime = False
            break
    
    if all_prime:
        print("✓ Verified: All values from n=0 to n=205 are prime!")
    else:
        # Count actual consecutive
        consecutive = generator.get_consecutive_primes(0, 300)
        print(f"Actual consecutive primes: {consecutive}")


if __name__ == "__main__":
    # Run all examples
    examples = [
        example_1_basic_generation,
        example_2_consecutive_primes,
        example_3_density_analysis,
        example_4_gap_patterns,
        example_5_large_primes,
        example_6_polynomial_ranges,
        example_7_verification
    ]
    
    try:
        for example in examples:
            example()
            print("\n" + "="*60 + "\n")
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
    
    # Keep the window open
    input("\nPress Enter to exit...")