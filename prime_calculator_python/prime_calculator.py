import math
import numpy as np
from multiprocessing import Pool

def small_primes_sieve(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    return np.nonzero(is_prime)[0]

def sieve_segment(low, high, small_primes):
    segment = np.ones(high - low + 1, dtype=bool)
    for prime in small_primes:
        start = max(prime*prime, (low + prime - 1) // prime * prime)
        if start > high:
            break
        segment[start - low:high - low + 1:prime] = False
    return np.nonzero(segment)[0] + low

def find_primes(limit, segment_size=1000000):
    sqrt_limit = int(math.sqrt(limit))
    small_primes = small_primes_sieve(sqrt_limit)
    primes = small_primes.tolist()

    segments = [(low, min(low + segment_size - 1, limit), small_primes) for low in range(sqrt_limit + 1, limit + 1, segment_size)]
    
    with Pool() as pool:
        for result in pool.starmap(sieve_segment, segments):
            primes.extend(result)

    return primes

def main():
    TARGET_PRIMES = 1000000
    LIMIT = 15485863

    primes = find_primes(LIMIT)

    if len(primes) < TARGET_PRIMES:
        print("Error: Not enough primes found. Increase the limit.")
        return

    # Write the first 1,000,000 prime numbers to a file
    with open("primes.txt", "w") as outfile:
        for i in range(TARGET_PRIMES):
            outfile.write(f"{i + 1}. {primes[i]}\n")

if __name__ == "__main__":
    main()
