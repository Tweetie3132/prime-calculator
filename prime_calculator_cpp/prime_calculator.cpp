#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

// Function to generate prime numbers using the Sieve of Eratosthenes
void segmentedSieve(int limit, std::vector<int> &primes) {
    int sqrtLimit = std::sqrt(limit);
    std::vector<bool> isPrime(sqrtLimit + 1, true);

    for (int i = 2; i * i <= sqrtLimit; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= sqrtLimit; j += i) {
                isPrime[j] = false;
            }
        }
    }

    std::vector<int> smallPrimes;
    for (int i = 2; i <= sqrtLimit; ++i) {
        if (isPrime[i]) {
            smallPrimes.push_back(i);
        }
    }

    int segmentSize = sqrtLimit;
    std::vector<bool> segment(segmentSize);

    #pragma omp parallel for schedule(dynamic)
    for (int low = sqrtLimit + 1; low <= limit; low += segmentSize) {
        std::fill(segment.begin(), segment.end(), true);

        int high = std::min(low + segmentSize - 1, limit);

        for (int prime : smallPrimes) {
            int start = std::max(prime * prime, (low + prime - 1) / prime * prime);
            for (int j = start; j <= high; j += prime) {
                segment[j - low] = false;
            }
        }

        #pragma omp critical
        {
            for (int i = low; i <= high; ++i) {
                if (segment[i - low]) {
                    primes.push_back(i);
                }
            }
        }
    }

    primes.insert(primes.begin(), smallPrimes.begin(), smallPrimes.end());
}

int main() {
    const int TARGET_PRIMES = 1000000;
    const int LIMIT = 15485863; // Approximate upper bound to ensure 1,000,000 primes
    std::vector<int> primes;

    auto start = std::chrono::high_resolution_clock::now();

    segmentedSieve(LIMIT, primes);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    if (primes.size() < TARGET_PRIMES) {
        std::cerr << "Error: Not enough primes found. Increase the limit." << std::endl;
        return 1;
    }

    // Write the first 1,000,000 prime numbers to a file
    std::ofstream outfile("primes.txt");
    for (int i = 0; i < TARGET_PRIMES; ++i) {
        outfile << i + 1 << ". " << primes[i] << std::endl;
    }
    outfile.close();

    return 0;
}
