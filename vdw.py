import cupy as cp
import time
import numpy as np
import sympy
import json
import multiprocessing as mp
import queue
from pathlib import Path
from typing import Optional
from math import log2


# --------------------------------------------------
# Global Parameters
# --------------------------------------------------

SAVE_PATH = "vdw_24b.json"
CHECKPOINT_STEPS = 10_000
MAX_RETRIES = 3
RANGE_LO = 1_000
RANGE_HI = 24_000_000_000

JACOBI_CACHE_SIZE = 256
PROG_LEVELS = [i / 32 for i in range(1, 32 + 1)]

MAX_K = 32

GEN_THREADS = 256
NUM_SCAN_BLOCKS = 1024

cuda_source = r"""
extern "C" {

enum SearchState { FIRST_RUN, LAST_RUN, SEARCH, VERIFY_BACK, SCAN_FWD, DONE };

constexpr int JACOBI_CACHE_SIZE = """ + str(JACOBI_CACHE_SIZE) + """;

__device__ inline bool calc_symbol(unsigned long long a, unsigned long long n, signed int* jacobi_cache) {
    unsigned int flips = 0;

    while ((a | n) >> 32 != 0) {
        int nbits = __ffsll(a) - 1;
        a >>= nbits;
        flips += (nbits & 1) & ((n >> 1) ^ (n >> 2));

        unsigned long long next_a = (a > n) ? a : n;
        unsigned long long next_n = (a < n) ? a : n;
        flips += (a < n) & ((next_a & next_n) >> 1);
        a = next_a - next_n;
        n = next_n;
    }

    unsigned int a32 = (unsigned int)a;
    unsigned int n32 = (unsigned int)n;
    while (n32 >= JACOBI_CACHE_SIZE || a32 >= JACOBI_CACHE_SIZE) {
        int nbits = __ffs(a32) - 1;
        a32 >>= nbits;
        flips += (nbits & 1) & ((n32 >> 1) ^ (n32 >> 2));

        unsigned int next_a = (a32 > n32) ? a32 : n32;
        unsigned int next_n = (a32 < n32) ? a32 : n32;
        flips += (a32 < n32) & ((next_a & next_n) >> 1);
        a32 = next_a - next_n;
        n32 = next_n;
    }

    unsigned int idx = a32 * JACOBI_CACHE_SIZE + n32;
    unsigned int bit_from_cache = (jacobi_cache[idx >> 5] >> (idx & 31)) & 1u;
    return !((flips & 1) ^ bit_from_cache);
}

__global__ void compute_and_scan_residues_merged(
    const unsigned long long p,
    const unsigned long long lo,
    const unsigned long long hi,
    const unsigned long long tasksize,
    int* block_max_ones, int* block_max_zeros,
    int* block_prefix_val, int* block_prefix_len,
    int* block_suffix_val, int* block_suffix_len,
    signed int* jacobi_cache, const int N) 
{   
    __shared__ int s_max_1[1024], s_max_0[1024], s_first_val[1024], s_first_len[1024];
    __shared__ int s_suffix_val[1024], s_suffix_len[1024], s_total_len[1024];
    __shared__ bool s_is_all_one[1024];

    unsigned long long startnum = lo + tasksize * ((unsigned long long)blockDim.x * blockIdx.x + threadIdx.x);
    unsigned long long endnum = startnum + tasksize - 1;
    if (endnum > hi) endnum = hi;
    if (startnum == 0) startnum++;

    int t_total_len = (startnum <= endnum) ? (int)(endnum - startnum + 1) : 0;
    SearchState state = (t_total_len > 0) ? FIRST_RUN : DONE;
    
    unsigned long long candidate = startnum;
    unsigned long long phase_start = startnum, phase_end = endnum;
    unsigned long long anchor_idx = 0, run_start = 0;
    int first_val = -1, first_len = 0, suffix_val = -1, suffix_len = 0, max_1 = 0, max_0 = 0;
    int current_bit = -1;

    bool is_active = (state != DONE);

    while (__any_sync(0xFFFFFFFF, is_active)) {
        // Inactive threads may have startnum >= p; a multiple of p never terminates in calc_symbol
        bool bit = calc_symbol(is_active ? candidate : 1, p, jacobi_cache);

        if (is_active) {
            switch (state) {
                case FIRST_RUN:
                    if (first_val == -1) { first_val = (int)bit; first_len = 1; candidate++; }
                    else if (bit == (bool)first_val) { first_len++; candidate++; }
                    else {
                        if (first_val == 1) max_1 = first_len; else max_0 = first_len;
                        phase_start = candidate; 
                        state = LAST_RUN; 
                        candidate = endnum;
                    }
                    if (candidate > endnum && state == FIRST_RUN) {
                        suffix_val = first_val; suffix_len = first_len;
                        if (first_val == 1) max_1 = first_len; else max_0 = first_len;
                        state = DONE;
                    }
                    break;

                case LAST_RUN:
                    if (suffix_val == -1) { suffix_val = (int)bit; suffix_len = 1; candidate--; }
                    else if (bit == (bool)suffix_val) { suffix_len++; candidate--; }
                    else {
                        if (suffix_val == 1) max_1 = (max_1 > suffix_len) ? max_1 : suffix_len; 
                        else max_0 = (max_0 > suffix_len) ? max_0 : suffix_len;
                        phase_end = candidate;
                        
                        // SETUP SEARCH
                        anchor_idx = phase_start;
                        candidate = anchor_idx + (N - 1);
                        state = (candidate <= phase_end) ? SEARCH : DONE;
                        
                        // Edge case: middle is smaller than N
                        if (candidate > phase_end && anchor_idx <= phase_end) {
                            state = SCAN_FWD;
                            run_start = anchor_idx;
                            current_bit = -1; // Flag to initialize on next loop
                            candidate = anchor_idx; 
                        }
                    }
                    if (candidate < phase_start && state == LAST_RUN) {
                        // Suffix run reaches phase_start: range is exactly two runs
                        if (suffix_val == 1) max_1 = (max_1 > suffix_len) ? max_1 : suffix_len;
                        else max_0 = (max_0 > suffix_len) ? max_0 : suffix_len;
                        state = DONE;
                    }
                    break;

                case SEARCH:
                    // We just jumped. The bit here dictates the ONLY run that could span the window.
                    current_bit = (int)bit;
                    candidate = anchor_idx + (N - 2); // Start scanning BACKWARDS
                    state = VERIFY_BACK;
                    break;

                case VERIFY_BACK:
                    if (bit != (bool)current_bit) {
                        // Mismatch found! Run is broken at 'candidate'.
                        // The earliest a new run could start is candidate + 1.
                        anchor_idx = candidate + 1;
                        candidate = anchor_idx + (N - 1);
                        
                        if (candidate > phase_end) {
                            // Tail end: Linear scan remainder
                            if (anchor_idx <= phase_end) {
                                state = SCAN_FWD;
                                run_start = anchor_idx;
                                current_bit = -1;
                                candidate = anchor_idx;
                            } else state = DONE;
                        } else {
                            state = SEARCH;
                        }
                    } else if (candidate <= anchor_idx) {
                        // Successfully verified backwards to the anchor. Run confirmed!
                        run_start = anchor_idx;
                        candidate = anchor_idx + N; 
                        state = SCAN_FWD;
                    } else {
                        candidate--; // Move backwards
                    }
                    break;

                case SCAN_FWD:
                    if (current_bit == -1) {
                        // Initialization for remainder tail scanning
                        current_bit = (int)bit;
                        candidate++;
                    } else if (bit != (bool)current_bit || candidate > phase_end) {
                        int total = (int)(candidate - run_start);
                        if (current_bit == 1) max_1 = (max_1 > total) ? max_1 : total; 
                        else max_0 = (max_0 > total) ? max_0 : total;
                        
                        if (candidate > phase_end) {
                            state = DONE;
                        } else {
                            // Run broke. Set anchor to the break point and jump.
                            anchor_idx = candidate;
                            candidate = anchor_idx + (N - 1);
                            
                            if (candidate > phase_end) {
                                state = SCAN_FWD;
                                run_start = anchor_idx;
                                current_bit = -1;
                                candidate = anchor_idx;
                            } else {
                                state = SEARCH;
                            }
                        }
                    } else {
                        candidate++;
                    }
                    break;
                
                case DONE:
                    break;
            }
        }
        is_active = (state != DONE);
    }

    int tid = threadIdx.x;
    s_max_1[tid] = max_1; s_max_0[tid] = max_0;
    s_first_val[tid] = first_val; s_first_len[tid] = first_len;
    s_suffix_val[tid] = suffix_val; s_suffix_len[tid] = suffix_len;
    s_total_len[tid] = t_total_len;
    s_is_all_one[tid] = (first_len == t_total_len && t_total_len > 0);

    __syncthreads();
    
    if (tid == 0) {
        int A_max_1 = 0, A_max_0 = 0, A_first_val = -1, A_first_len = 0;
        int A_suffix_val = -1, A_suffix_len = 0, A_total_len = 0;
        bool A_is_all_one = true;

        for (int i = 0; i < blockDim.x; i++) {
            int B_tot = s_total_len[i];
            if (B_tot == 0) continue;

            if (A_total_len == 0) {
                A_max_1 = s_max_1[i]; A_max_0 = s_max_0[i];
                A_first_val = s_first_val[i]; A_first_len = s_first_len[i];
                A_suffix_val = s_suffix_val[i]; A_suffix_len = s_suffix_len[i];
                A_is_all_one = s_is_all_one[i]; A_total_len = B_tot;
            } else {
                A_max_1 = (A_max_1 > s_max_1[i]) ? A_max_1 : s_max_1[i];
                A_max_0 = (A_max_0 > s_max_0[i]) ? A_max_0 : s_max_0[i];

                if (A_suffix_val == s_first_val[i] && A_suffix_val != -1) {
                    int merged_len = A_suffix_len + s_first_len[i];
                    if (A_suffix_val == 1) A_max_1 = (A_max_1 > merged_len) ? A_max_1 : merged_len;
                    else A_max_0 = (A_max_0 > merged_len) ? A_max_0 : merged_len;

                    if (A_is_all_one) A_first_len = merged_len;
                    if (s_is_all_one[i]) A_suffix_len = merged_len;
                    else { A_suffix_val = s_suffix_val[i]; A_suffix_len = s_suffix_len[i]; }
                    A_is_all_one = (A_is_all_one && s_is_all_one[i]);
                } else {
                    A_is_all_one = false;
                    A_suffix_val = s_suffix_val[i]; A_suffix_len = s_suffix_len[i];
                }
                A_total_len += B_tot;
            }
        }
        block_max_ones[blockIdx.x] = A_max_1; block_max_zeros[blockIdx.x] = A_max_0;
        block_prefix_val[blockIdx.x] = A_first_val; block_prefix_len[blockIdx.x] = A_first_len;
        block_suffix_val[blockIdx.x] = A_suffix_val; block_suffix_len[blockIdx.x] = A_suffix_len;
    }
}
}
"""

def fmt_secs(s):
    w, s = divmod(int(s), 604800); d, s = divmod(s, 86400)
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return ", ".join(f"{v}{u}" for v, u in [(w,"w"),(d,"d"),(h,"h"),(m,"m"),(s,"s")] if v) or "0s"

module = cp.RawModule(code=cuda_source)

def calc_progressions(p: int, metadata: cp.ndarray, scaler: float, lastscaler: Optional[float] = None, first_pre_l: Optional[int] = None, jacobi_cache: cp.ndarray = None, skip=None):
    lo = 1 if lastscaler is None else int(p // 2 * lastscaler)
    lo = max(1, lo - 100)
    hi = int(p // 2 * scaler) + 100

    total_elements = hi - lo + 1
    total_threads = NUM_SCAN_BLOCKS * GEN_THREADS
    tasksize = max(1, (total_elements + total_threads - 1) // total_threads)

    merged_kern = module.get_function('compute_and_scan_residues_merged')
    cp.cuda.Device().synchronize()

    merged_kern((NUM_SCAN_BLOCKS,), (GEN_THREADS,), (
        p, lo, hi, tasksize,
        metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], metadata[5], jacobi_cache, skip
    ))

    h_meta = metadata.get()
    cp.cuda.Device().synchronize()

    res_m1, res_m0 = int(np.max(h_meta[0])), int(np.max(h_meta[1]))
    pre_v, pre_l, suf_v, suf_l = h_meta[2], h_meta[3], h_meta[4], h_meta[5]

    current_first_pre_l = first_pre_l
    if lastscaler is None:
        current_first_pre_l = int(pre_l[0])

    matches = (suf_v[:-1] == pre_v[1:])
    if np.any(matches):
        comb_lens = suf_l[:-1][matches] + pre_l[1:][matches]
        comb_vals = suf_v[:-1][matches]
        res_m1 = max(res_m1, int(np.max(comb_lens[comb_vals == 1], initial=0)))
        res_m0 = max(res_m0, int(np.max(comb_lens[comb_vals == 0], initial=0)))

    if scaler > 0.99 and current_first_pre_l is not None:
        if p % 4 == 1:
            res_m1 = max(res_m1, int(current_first_pre_l * 2 + 1))
        else:
            res_m1 = max(res_m1, int(current_first_pre_l + 1))

    return (res_m1, res_m0), current_first_pre_l


# --------------------------------------------------
# GPU worker
# --------------------------------------------------
def gpu_worker(gpu_id: int, prime_q: mp.Queue, result_q: mp.Queue):
    import numpy as np

    def compute_jacobi(a, n):
        from math import gcd
        a, n = int(a), int(n)
        if n == 0:
            return a not in (1, -1)
        if gcd(a, n) != 1:
            return True
        flips = 0
        while a != 0:
            nbits = (a & -a).bit_length() - 1
            a >>= nbits
            if nbits & 1:
                flips ^= (n >> 1) ^ (n >> 2)
            if a < n:
                flips ^= (a & n) >> 1
                a, n = n - a, a
            else:
                a -= n
        return flips & 1

    jacobi_cache = np.zeros((JACOBI_CACHE_SIZE, JACOBI_CACHE_SIZE), dtype=np.int8)
    for a in range(JACOBI_CACHE_SIZE):
        for n in range(JACOBI_CACHE_SIZE):
            jacobi_cache[a, n] = compute_jacobi(a, n)
    jacobi_cache = np.packbits(jacobi_cache, bitorder='little').view(np.uint32)

    cp.cuda.Device(gpu_id).use()
    jacobi_cache = cp.array(jacobi_cache)

    while True:
        try:
            item = prime_q.get(timeout=2)
        except queue.Empty:
            continue
        if item is None:
            break

        p, best_snapshot = item
        try:
            metadata = cp.zeros((6, NUM_SCAN_BLOCKS), dtype=cp.int32)
            global_m1 = global_m0 = 0
            last_scaler = None
            first_pre_l = None
            best_m = 0

            skip = max(i for i in range(len(best_snapshot)) if best_snapshot[i] == 0)
            for scaler in PROG_LEVELS:
                (m1, m0), first_pre_l = calc_progressions(p, metadata, scaler, last_scaler, first_pre_l, jacobi_cache, skip)
                global_m1 = max(global_m1, m1)
                global_m0 = max(global_m0, m0)
                best_m = max(global_m1, global_m0)
                last_scaler = scaler

                if best_m < len(best_snapshot) and p < best_snapshot[best_m]:
                    break

            if best_m <= skip and not (best_m < len(best_snapshot) and p < best_snapshot[best_m]):
                print(f"Failed to skip {best_m=} < {skip=}")
                last_scaler = None
                for scaler in PROG_LEVELS:
                    (m1, m0), first_pre_l = calc_progressions(p, metadata, scaler, last_scaler, first_pre_l, jacobi_cache, 4)
                    global_m1 = max(global_m1, m1)
                    global_m0 = max(global_m0, m0)
                    best_m = max(global_m1, global_m0)
                    last_scaler = scaler

                    if best_m < len(best_snapshot) and p < best_snapshot[best_m]:
                        break

            result_q.put(("ok", int(p), best_m))
        except Exception as e:
            result_q.put(("err", int(p), repr(e)))

    result_q.put(("exited", gpu_id))


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    file_path = Path(SAVE_PATH)

    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        saved_commit, start_hi, best_primes = data
        next_to_dispatch = sympy.prevprime(int(saved_commit))
        commit_prime = int(saved_commit)
        init = saved_commit
    else:
        start_hi = RANGE_HI
        next_to_dispatch = sympy.prevprime(start_hi)
        best_primes = [0] * MAX_K
        commit_prime = start_hi
        init = start_hi

    last_checkpoint = commit_prime
    ngpus = max(1, cp.cuda.runtime.getDeviceCount())
    prime_q = mp.Queue(maxsize=4 * ngpus)
    result_q = mp.Queue()

    workers = [mp.Process(target=gpu_worker, args=(gid, prime_q, result_q), daemon=True) for gid in range(ngpus)]
    for w in workers:
        w.start()

    in_flight_count = 0
    pending_finished = set()
    retries = {}
    start_time = time.time()
    last_print = time.time()

    def do_checkpoint(commit_val):
        with open(file_path, 'w') as f:
            json.dump([int(commit_val), int(start_hi), best_primes], f)

    while True:
        while not prime_q.full() and next_to_dispatch > RANGE_LO:
            prime_q.put((int(next_to_dispatch), list(best_primes)))
            in_flight_count += 1
            next_to_dispatch = sympy.prevprime(int(next_to_dispatch))

        try:
            msg = result_q.get(timeout=1.0)
        except queue.Empty:
            if next_to_dispatch <= RANGE_LO and in_flight_count == 0:
                break
            continue

        tag = msg[0]
        if tag == "ok":
            _, p_finished, best_m = msg

            for L in range(best_m, len(best_primes)):
                best_primes[L] = max(best_primes[L], p_finished)

            in_flight_count -= 1
            pending_finished.add(p_finished)

            while True:
                prev_p = sympy.prevprime(commit_prime)
                if prev_p in pending_finished:
                    pending_finished.remove(prev_p)
                    commit_prime = int(prev_p)
                    if last_checkpoint > commit_prime + CHECKPOINT_STEPS:
                        do_checkpoint(commit_prime)
                        last_checkpoint = commit_prime
                else:
                    break

            if time.time() - last_print > 10.0:
                elapsed = time.time() - start_time
                etr = int((commit_prime - RANGE_LO) / (init - commit_prime + 1) * elapsed) if init != commit_prime else 0
                print(f"CP={commit_prime} | Elapsed={fmt_secs(elapsed)} | ETR={fmt_secs(etr)} | InFlight={in_flight_count}")
                print(f"Best: {best_primes}")
                last_print = time.time()

        elif tag == "err":
            _, p_err, exc = msg
            p_err = int(p_err)
            retries[p_err] = retries.get(p_err, 0) + 1
            print(f"worker error on p={p_err} (attempt {retries[p_err]}): {exc}")
            if retries[p_err] > MAX_RETRIES:
                print(f"Giving up on p={p_err}; stopping so the checkpoint never passes it.")
                break
            prime_q.put((p_err, list(best_primes)))
        elif tag == "exited":
            in_flight_count = 0

        if next_to_dispatch <= RANGE_LO and in_flight_count == 0:
            break

    for _ in workers:
        prime_q.put(None)
    for w in workers:
        w.join(timeout=5)

    do_checkpoint(commit_prime)
    print("Done. Final best_primes =", best_primes)
    print(f"Time elapsed: {time.time() - start_time}")