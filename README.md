# Lower bounds for van der Waerden numbers based on quadratic residues

Fast GPU implementation of the two-color Rabung construction for colorings based on quadratic residues that avoid large monochromatic arithmetic progressions.

## Scripts

This repository contains two scripts, `vdw.py` and `vdw_scatter.py`. The former is the faster "direct" version that supports any prime encountered in practice and the latter is the slower scattering version that only supports primes up to around 8 billion. Each supports dispatching primes to multiple GPUs, and contains several in-code constants. `SAVE_PATH` and `CHECKPOINT_STEPS` control where and how often the best primes so far are checkpointed, and `RANGE_LO` and `RANGE_HI` control the range of primes tested.

The checkpoint records an array `best_primes` where `best_primes[L]` is the largest prime whose longest run of consecutive residues or non-residues is at most L, so the bound is 
w(k;2)>(k−1)⋅best_primes[k−1]+1

Simply `pip install -r requirements.txt` in your environment then run `nohup python vdw.py &`.

## Results
Running `vdw.py` on primes up to 24 billion took around four weeks on four RTX 3090 GPUs whereas a distributed computing project with hundreds of volunteers took over a year to do the same computation for primes up to 24 billion. Our results match theirs for the k = 8 through 24:

`[0, 0, 0, 0, 0, 0, 0, 1069, 3389, 11497, 17863, 58013, 136859, 239873, 608789, 1091339, 2899861, 5357603, 13919273, 27700919, 70483537, 122954173, 282097363, 477395357, 1138900957, 2125065391, 4195501393, 8758090373, 20968447439, 23999633509, 23999998277, 23999998769]`