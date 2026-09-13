# Lower bounds for van der Waerden numbers based on quadratic residues

Fast GPU implementation of the two-color Rabung construction for colorings based on quadratic residues that avoid large monochromatic arithmetic progressions.

## Scripts

This repository contains two scripts, `vdw.py` and `vdw_scatter.py`. The former is the faster "direct" version that supports any prime encountered in practice and the latter is the slower scattering version that only supports primes up to around 8 billion. Each supports dispatching primes to multiple GPUs, and contains several in-code constants. `SAVE_PATH` and `CHECKPOINT_STEPS` control where and how often the best primes so far are checkpointed, and `RANGE_LO` and `RANGE_HI` control the range of primes tested.

The checkpoint records an array `best_primes` where `best_primes[L]` is the largest prime whose longest run of consecutive residues or non-residues is at most L, so the bound is 
w(k;2)>(k−1)⋅best_primes[k−1]+1

Simply `pip install -r requirements.txt` in your environment then run `nohup python vdw.py &`.