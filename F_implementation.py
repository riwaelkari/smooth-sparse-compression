import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
rng = np.random.default_rng(12345)    #random number generator with fixed seed for reproducibility
import matplotlib.pyplot as plt 
import pandas as pd

# ------------------------------
# Utilities
# ------------------------------
def H2(eps: float) -> float:
    """Binary entropy in bits."""
    if eps <= 0 :  # avoid log(0)
        eps = 1e-12
    elif eps >= 1:  # avoid log(1-1=0)
        eps = 1 - 1e-12
    return -(eps*np.log2(eps) + (1-eps)*np.log2(1-eps))

def bernoulli_eps(n: int, eps: float) -> np.ndarray:
    """Generate n bits from Bernoulli(eps) distribution."""
    # Step 1: generate n uniform random floats in [0, 1)
    random_values = rng.random(n)

    # Step 2: compare each to eps -> array of True/False
    bernoulli_bools = random_values < eps

    # Step 3: convert True/False to 1/0 integers (uint8 since we only need 0/1)
    bernoulli_bits = bernoulli_bools.astype(np.uint8)

    return bernoulli_bits


def bits_to_bytestr(b: np.ndarray) -> bytes:
    """
    Convert an array of bits (0/1 integers) into a compact bytes object.
    
    """

    # Step 1: Ensure the array is of dtype uint8 (required for np.packbits)
    b_uint8 = b.astype(np.uint8)

    # Step 2: Pack every 8 bits into a single byte.
    # bitorder='little' means b[0] is the least significant bit in the first byte.
    packed = np.packbits(b_uint8, bitorder='little')

    # Step 3: Convert the packed NumPy array to a raw bytes object.
    # This makes it hashable and easy to store in sets or dicts.
    byte_string = packed.tobytes()

    return byte_string


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Calculate the Hamming distance between two binary arrays a and b.
    """
    return int(np.count_nonzero(a ^ b))  #binary -> XOR and count non-zero bits

# ------------------------------
# Build sparse linear graph H (parity-check-like rows)
# ------------------------------
def build_linear_graph(n: int, m: int, var_deg: int = 3) -> List[np.ndarray]:
    """
    Return list-of-arrays rows: row a lists variable indices used by parity row a.
    Variable degrees are ~ var_deg; check degrees are balanced automatically.
    """
    E = n * var_deg  # total edges
    # distribute edges across m rows as evenly as possible
    base = E // m
    rows_with_plus1 = E - base * m
    check_degs = np.full(m, base, dtype=int)
    check_degs[:rows_with_plus1] += 1
    # variable stubs
    stubs = np.repeat(np.arange(n, dtype=int), var_deg)
    rng.shuffle(stubs)
    rows = []
    start = 0
    for a in range(m):
        d = check_degs[a]
        take = stubs[start:start+d]
        start += d
        # ensure no duplicates in one row (if duplicates, fix by small resampling)
        if len(set(take.tolist())) != d:
            # simple de-dup repair
            seen = set()
            fixed = []
            pool = set(range(n))
            for v in take:
                if v in seen:
                    # pick a replacement not yet used in row
                    candidate = rng.choice(list(pool - seen))
                    fixed.append(candidate)
                    seen.add(candidate)
                else:
                    fixed.append(v)
                    seen.add(v)
            take = np.array(fixed, dtype=int)
        rows.append(np.array(take, dtype=int))
    return rows

def L_apply(rows: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    z = np.zeros(len(rows), dtype=np.uint8)
    for a, idxs in enumerate(rows):
        z[a] = np.bitwise_xor.reduce(x[idxs]) if len(idxs) else 0
    return z

# ------------------------------
# Build nonlinear graph and f
# ------------------------------
def pattern_probs(eps: float, k: int) -> np.ndarray:
    """
    Compute the probability of each k-bit pattern (0..2^k-1)
    under an i.i.d. Bernoulli(eps) source.

    """
    num_patterns = 1 << k  # 2^k
    probs = np.zeros(num_patterns, dtype=float)
    for idx in range(num_patterns):
        ones = bin(idx).count("1")
        probs[idx] = (eps ** ones) * ((1 - eps) ** (k - ones))
    return probs


def simple_f_table_weighted(eps, k):
    """
    Build a Boolean function f table of length 2^k
    so that P[f=1] ≈ 0.5 for X~Bernoulli(eps)^k.

    """
    num_patterns = 1 << k  # 2^k patterns

    # 1) Compute probabilities of each pattern under Bernoulli(eps)
    probs = pattern_probs(eps, k)

    # 2) Sort patterns by probability (most likely first) (sorts indices based on values of probabilities)
    order = np.argsort(-probs)

    # 3) Greedily assign '1' to patterns until total probability <= 0.5
    f_table = np.zeros(num_patterns, dtype=np.uint8)
    total_prob = 0.0
    for idx in order:
        if total_prob + probs[idx] <= 0.5:
            f_table[idx] = 1
            total_prob += probs[idx]

    # Done — f_table now outputs 1 for chosen patterns, 0 otherwise
    return f_table


def eval_f_on_block(block_bits: np.ndarray, f_table: np.ndarray) -> np.uint8:
    """Convert block_bits (little-endian) to int and look up in f_table."""
    k = len(block_bits)
    idx = 0
    for i in range(k):
        bit_value = int(block_bits[i]) & 1     # check if it's either 0 or 1
        shifted_bit = bit_value << i           # Move it to position i (in binary)
        idx |= shifted_bit                     # OR it into idx
    # idx is now the integer representation of the block_bits (ex: 101 -> idx=5)

    return f_table[idx]

def build_nonlinear_graph(n: int, m: int, k: int = 6, r: int = 4) -> List[List[np.ndarray]]:
    """
    For each output coordinate, choose r random blocks of k distinct variable indices.
    No degree balancing.
    """
    a_blocks = []
    for _ in range(m):
        blocks = []
        for _ in range(r):
            # Pick k distinct variable indices from [0, n)
            block = np.array(rng.choice(n, size=k, replace=False), dtype=int)   #each block is a random selection of k indices
            blocks.append(block)
        a_blocks.append(blocks)
    return a_blocks

def N_apply(blocks: List[List[np.ndarray]], x: np.ndarray, f_table: np.ndarray) -> np.ndarray:
    m = len(blocks)                      # number of outputs
    z = np.zeros(m, dtype=np.uint8)      # output vector

    a = 0                                # manual index counter
    for blist in blocks:                 # loop over each output's block list
        v = 0
        for B in blist:                   # loop over each block in the block list
            f_val = int(eval_f_on_block(x[B], f_table))
            v ^= f_val                    # XOR into accumulator v
        z[a] = v
        a += 1                            # increment index manually

    return z

# ------------------------------
# Full encoder F = L ⊕ N
# ------------------------------
class SmoothSparseEncoder:
    def __init__(self, n: int, m: int, eps: float,
                 var_deg_linear: int = 3, k: int = 6, r: int = 4):
        """
        n: blocklength
        m: compressed length
        eps: Bernoulli(1) probability of source symbols
        var_deg_linear: variable node degree (linear graph)
        k: nb of inputs of the non-linear node f
        r: how many k-blocks each output sums over
        """
        self.n = n
        self.m = m
        self.eps = eps
        self.rows_H = build_linear_graph(n, m, var_deg=var_deg_linear)
        self.blocks = build_nonlinear_graph(n, m, k=k, r=r)
        self.k = k
        self.r = r
        self.f_table = simple_f_table_weighted(eps, k)


        # Precompute an estimate of P[f=1] under Bernoulli(eps) for sanity 
        probs = pattern_probs(eps, k)
        p1 = float((self.f_table * probs).sum())
        self.f_bias = abs(p1 - 0.5)    #check how close to 0.5 we are

    def F(self, x: np.ndarray) -> np.ndarray:
        """Compute z = F(x) = L(x) XOR N(x)."""
        z_lin = L_apply(self.rows_H, x)
        z_non = N_apply(self.blocks, x, self.f_table)
        return z_lin ^ z_non

    # Diagnostics
    def empirical_smoothness(self, S: int = 200, flip_per_x: int = 1) -> Tuple[float, int]:
        """
        Estimate L-smoothness: average and max # of output bits that change when flipping one input bit (Lavg and Lmax).
        """
        changes = []
        for _ in range(S):
            x = bernoulli_eps(self.n, self.eps)   # Generate a random input vector
            z = self.F(x)                         # Encode it

            for _ in range(flip_per_x):
                i = rng.integers(0, self.n)       # Pick a random position in x
                x2 = x.copy()                     # Make a copy of x
                x2[i] ^= 1                        # Flip that bit (XOR with 1)
                z2 = self.F(x2)                    # Encode the flipped input
                changes.append(hamming(z, z2))    # Count differing bits in output
            changes = np.array(changes)
            Lavg = float(changes.mean())
            Lmax = int(changes.max())
            return Lavg, Lmax

    def collision_rate(self, S: int = 5000) -> Tuple[int, int, float]:
        """
        Draw S random 'typical' x's and count how often F(x) maps to the same output.

        """
        seen = {}       # Dictionary to store outputs we've already seen (key = bytes of z)
        collisions = 0  

        for _ in range(S):
            # 1. Generate a random input vector from Bernoulli(eps) distribution
            x = bernoulli_eps(self.n, self.eps)

            # 2. Encode it using function F
            z = self.F(x)

            # 3. Convert binary output array to a compact bytes object
            key = bits_to_bytestr(z)

            # 4. Check if this output was already produced
            if key in seen:
                #it's a collision
                collisions += 1
            else:
                # First time seeing this output → store it
                seen[key] = 1

        # 5. Count unique outputs (size of dictionary)
        unique = len(seen)

        # 6. Compute collision rate as fraction of all samples that were collisions
        #    (max(1, S) avoids division by zero if S=0)
        return unique, collisions, collisions / max(1, S)


# ------------------------------
# Helper: sweep rates near entropy, quick experiment
# ------------------------------
def quick_sweep(n=512, eps=0.12, overheads=(0.30, 0.20, 0.12, 0.08, 0.06, 0.04),
                var_deg_linear=3, k=6, r=4, samples=4000):
    """
    Try several rates R = H2(eps) + overhead and report collisions + smoothness.
    """
    # 1) Compute the theoretical minimum bits per symbol for Bernoulli(eps) (entropy)
    baseH = H2(eps)

    # 2) Store results for each tested overhead
    results = []

    # 3) Loop over each chosen overhead value
    for ov in overheads:
        # 3a) Calculate the test rate = minimum + overhead
        R = baseH + ov

        # 3b) Convert the rate to an integer number of output bits (m)
        #     using ceiling to ensure enough capacity
        m = int(np.ceil(R * n))

        # 4) Build an encoder with these parameters
        enc = SmoothSparseEncoder(n, m, eps, var_deg_linear=var_deg_linear, k=k, r=r)

        # 5) Measure collisions:
        #    - uniq  = number of distinct codewords produced
        #    - colnb = number of collisions observed
        #    - colrate = fraction of samples that collide
        uniq, colnb, colrate = enc.collision_rate(S=samples)

        # 6) Measure smoothness:
        lavg, lmax = enc.empirical_smoothness(S=min(1000, samples//5))

        # 7) Save all metrics in a results dictionary
        results.append({
            "R_bits_per_sym": R,
            "m": m,
            "overhead_bits": ov,
            "f_bias_abs": enc.f_bias,
            "unique": uniq,
            "collisions": colnb,
            "collision_rate": colrate,
            "smooth_avg": lavg,
            "smooth_max": lmax
        })

        # 8) Print summary for this overhead
        print(f"R={R:.3f} (m={m}): collisions={colnb}/{samples} "
              f"({colrate:.3%}), L(avg,max)=({lavg:.2f},{lmax}), f_bias≈{enc.f_bias:.4f}")

    # 9) Return all collected results
    return results



# ------------------------------
# Example run
# ------------------------------

# ------------------------------
# Example run
# ------------------------------
if __name__ == "__main__":
    n = 512
    eps = 0.12
    print("Source entropy H2(eps):", H2(eps))

    results = quick_sweep(
        n=n, eps=eps,
        overheads=(0.30, 0.20, 0.12, 0.08, 0.06, 0.04),
        var_deg_linear=3, k=6, r=4,
        samples=8000  # stress test with more samples
    )

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)

    # ------------------------------
    # Two subplots in one figure
    # ------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Collision rate subplot
    axs[0].plot(df["R_bits_per_sym"], df["collision_rate"], marker="o", color="red", label="Collision Rate")
    axs[0].set_xlabel("Rate (bits/symbol)")
    axs[0].set_ylabel("Collision Rate")
    axs[0].set_title("Collision Rate vs. Rate")
    axs[0].grid(True)
    axs[0].legend()

    # Smoothness subplot
    axs[1].plot(df["R_bits_per_sym"], df["smooth_avg"], marker="o", label="Average Smoothness")
    axs[1].plot(df["R_bits_per_sym"], df["smooth_max"], marker="s", label="Max Smoothness")
    axs[1].set_xlabel("Rate (bits/symbol)")
    axs[1].set_ylabel("Smoothness (# changed output bits)")
    axs[1].set_title("Smoothness vs. Rate")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
