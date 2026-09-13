import cupy as cp
import time
import numpy as np
import sympy
import json
import multiprocessing as mp
import queue
from pathlib import Path
from typing import Optional

# --------------------------------------------------
# Global Parameters
# --------------------------------------------------

SAVE_PATH = "vdw_8b_scatter.json"
CHECKPOINT_STEPS = 10_000
MAX_RETRIES = 3
RANGE_LO = 1_000
RANGE_HI = 8_000_000_000
PROG_LEVELS = [32, 16, 8, 4, 2]
MAX_K = 32

# Kernel parameters
GEN_LENGTH = 128
GEN_THREADS = 256
NUM_SCAN_BLOCKS = 32768

# --------------------------------------------------
# CUDA Kernels
# --------------------------------------------------
cuda_source = r'''
extern "C" {
    __global__ void generate_residues_fast(const unsigned long long p, 
                                        const unsigned long long lo, 
                                        const unsigned long long hi,
                                        const unsigned long long tasksize,
                                        unsigned int* mask) {
        unsigned long long startnum = tasksize * ((unsigned long long)blockDim.x * blockIdx.x + threadIdx.x);
        unsigned long long endnum = min((p-1) / 2, startnum + tasksize);
        unsigned long long x = startnum;
        unsigned long long square = (x * x) % p;

        while (x <= endnum) {
            if (square <= hi && square >= lo) {
                atomicOr(&mask[square >> 5], (1u << (square & 31)));
            }
            square += 2 * x + 1;
            square -= p * (square >= p);
            x++;
        }
    }

    __global__ void scan_segments_limit_aware(const unsigned int* mask,
                                            const unsigned long long num_words,
                                            const unsigned long long lo,
                                            const unsigned long long hi,
                                            int* block_max_ones, int* block_max_zeros,
                                            int* block_prefix_val, int* block_prefix_len,
                                            int* block_suffix_val, int* block_suffix_len) {


        
        unsigned long long tid = blockIdx.x;
        unsigned long long words_per_block = (num_words + gridDim.x - 1) / gridDim.x;
        unsigned long long start_word = tid * words_per_block;
        unsigned long long end_word = min(start_word + words_per_block, num_words);
        end_word = min(end_word, (hi + 31LL) / 32LL);

        if (32 * end_word < lo)
            return;
        if (32 * start_word > hi)
            return;

        int max_1 = 0, max_0 = 0, curr_1 = 0, curr_0 = 0;
        int first_val = -1, first_len = 0;

        for (unsigned long long i = start_word; i < end_word; i++) {
            unsigned int word = mask[i];
            for (int b = i == 0; b < 32; b++) {
                int bit = (word >> b) & 1;
                if (bit == 1) {
                    curr_1++;
                    if (curr_0 > 0) {
                        if (first_val == -1) { first_val = 0; first_len = curr_0; }
                        if (curr_0 > max_0) max_0 = curr_0;
                        curr_0 = 0;
                    }
                } else {
                    curr_0++;
                    if (curr_1 > 0) {
                        if (first_val == -1) { first_val = 1; first_len = curr_1; }
                        if (curr_1 > max_1) max_1 = curr_1;
                        curr_1 = 0;
                    }
                }
            }
        }

        if (first_val == -1) {
            first_val = (curr_1 > 0) ? 1 : 0;
            first_len = (curr_1 > 0) ? curr_1 : curr_0;
        }

        if (curr_1 > max_1) max_1 = curr_1;
        if (curr_0 > max_0) max_0 = curr_0;
        
        block_max_ones[tid] = max_1;
        block_max_zeros[tid] = max_0;
        block_prefix_val[tid] = first_val;
        block_prefix_len[tid] = first_len;
        block_suffix_val[tid] = (curr_1 > 0) ? 1 : 0;
        block_suffix_len[tid] = (curr_1 > 0) ? curr_1 : curr_0;
    }
}   
'''

def fmt_secs(s):
    w, s = divmod(int(s), 604800)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return ", ".join(
        f"{v}{u}" for v, u in [(w,"w"),(d,"d"),(h,"h"),(m,"m"),(s,"s")] if v
    ) or "0s"

# --------------------------------------------------
# Core GPU calculation
# --------------------------------------------------
def calc_progressions(p: int, mask: cp.ndarray, metadata: cp.ndarray, divisor:int=2, lastdivisor:Optional[int]=None):
    limit = p // divisor
    assert p // 2 < 2 ** 32, f"Found overlarge prime {p}" 
    # Add at least 128 - 31 bits of extra space since consecutive residues/non-residues may loop over
    num_words = (limit >> 5) + 4
    limit = num_words * 32

    module = cp.RawModule(code=cuda_source)
    gen_kern = module.get_function('generate_residues_fast')
    scan_kern = module.get_function('scan_segments_limit_aware')

    gen_blocks = int(((p - 1) // 2 + GEN_THREADS - 1) // GEN_THREADS // GEN_LENGTH) + 1

    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    gen_kern((gen_blocks,), (GEN_THREADS,), (p, 0 if lastdivisor is None else p // lastdivisor, limit, GEN_LENGTH, mask))
    cp.cuda.Device().synchronize()
    gen_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    scan_kern((NUM_SCAN_BLOCKS,), (1,), (
        mask, ((p >> 6) + 4),
        0 if lastdivisor is None else p // lastdivisor, limit,
        metadata[0], metadata[1], metadata[2], metadata[3], metadata[4], metadata[5]
    ))
    h_meta = metadata.get()
    cp.cuda.Device().synchronize()
    scan_time = time.perf_counter() - t1

    # ---------------- deterministic stitch ----------------
    t2 = time.perf_counter()
    res_m1, res_m0 = int(np.max(h_meta[0])), int(np.max(h_meta[1]))

    if divisor == 2:
        pre_v, pre_l, suf_v, suf_l = h_meta[2], h_meta[3], h_meta[4], h_meta[5]
        
        matches = (suf_v[:-1] == pre_v[1:])
        if np.any(matches):
            comb_lens = suf_l[:-1][matches] + pre_l[1:][matches]
            comb_vals = suf_v[:-1][matches]
            m1_stitch = np.max(comb_lens[comb_vals == 1], initial=0)
            m0_stitch = np.max(comb_lens[comb_vals == 0], initial=0)
            res_m1 = max(int(res_m1), int(m1_stitch))
            res_m0 = max(int(res_m0), int(m0_stitch))

        if p % 4 == 1:
            res_m1 = max(res_m1, pre_l[0] * 2 + 1)
        else:
            res_m1 = max(res_m1, pre_l[0] + 1)
                
    stitch_time = time.perf_counter() - t2

    return (res_m1, res_m0), (gen_time, scan_time, stitch_time)

# --------------------------------------------------
# GPU worker
# --------------------------------------------------
def gpu_worker(gpu_id: int, prime_q: mp.Queue, result_q: mp.Queue):
    cp.cuda.Device(gpu_id).use()
    while True:
        try:
            item = prime_q.get(timeout=2)
        except queue.Empty:
            continue
        if item is None:
            break

        p, best_snapshot = item
        best_snapshot = list(best_snapshot)

        try:
            mask = cp.zeros((p >> 6) + 4, dtype=cp.uint32)
            metadata = cp.zeros((6, NUM_SCAN_BLOCKS), dtype=cp.int32)

            per_div = {}
            total_gen = total_scan = total_stitch = 0.0

            last_div = None
            for divisor in PROG_LEVELS:
                (m1, m0), (gt, st, ct) = calc_progressions(p, mask, metadata, divisor, last_div)
                m = max(m1, m0)

                per_div[divisor] = {"m": int(m), "gen": float(gt), "scan": float(st), "stitch": float(ct)}
                total_gen += gt
                total_scan += st
                total_stitch += ct
                last_div = divisor

                if m < len(best_snapshot) and p < best_snapshot[m]:
                    break

            result_q.put(("ok", int(p), per_div, (total_gen, total_scan, total_stitch)))

        except Exception as e:
            result_q.put(("err", int(p), repr(e)))

    result_q.put(("exited", gpu_id))

# --------------------------------------------------
# Main handler
# --------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    file_path = Path(SAVE_PATH)
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        saved_commit, start_hi, best_primes, phase_stats, prog_levels = data
        prog_levels = {int(k): v for k, v in prog_levels.items()}
        print(f"Loaded checkpoint at commit={saved_commit}, hi={start_hi}.")
        next_to_dispatch = sympy.prevprime(int(saved_commit))
        commit_prime = int(saved_commit)
        init = saved_commit
    else:
        start_hi = RANGE_HI
        next_to_dispatch = sympy.prevprime(start_hi)
        best_primes = [0] * MAX_K
        phase_stats = {"gen": 0.0, "scan": 0.0, "stitch": 0.0}
        prog_levels = dict.fromkeys(PROG_LEVELS, 0.0)
        commit_prime = start_hi
        init = start_hi

    last_checkpoint = commit_prime
    ngpus = max(1, cp.cuda.runtime.getDeviceCount())
    prime_q = mp.Queue(maxsize=4 * ngpus)
    result_q = mp.Queue()

    workers = [mp.Process(target=gpu_worker, args=(gid, prime_q, result_q), daemon=True)
            for gid in range(ngpus)]
    for w in workers: w.start()

    in_flight_count = 0
    pending_finished = set()
    retries = {}
    start = time.time()
    last_print = time.time()

    def do_checkpoint(commit_val):
        info = [int(commit_val), int(start_hi), best_primes, phase_stats, prog_levels]
        with open(file_path, 'w') as f:
            json.dump(info, f)

    while True:
        # dispatch primes
        while not prime_q.full() and next_to_dispatch > RANGE_LO:
            prime_q.put((int(next_to_dispatch), list(best_primes)))
            in_flight_count += 1
            next_to_dispatch = sympy.prevprime(int(next_to_dispatch))

        # collect results
        try:
            msg = result_q.get(timeout=2.0)
        except queue.Empty:
            if next_to_dispatch <= RANGE_LO and in_flight_count == 0:
                break
            continue

        tag = msg[0]
        if tag == "ok":
            _, p_finished, per_div, (gtime, stime, ctime) = msg
            p_finished = int(p_finished)
            phase_stats['gen'] += float(gtime)
            phase_stats['scan'] += float(stime)
            phase_stats['stitch'] += float(ctime)

            for div, info in per_div.items():
                prog_levels[div] = prog_levels.get(div, 0.0) + float(info.get('gen',0.0) + info.get('scan',0.0) + info.get('stitch',0.0))

            best_m = max(info['m'] for info in per_div.values())
            for L in range(best_m, len(best_primes)):
                best_primes[L] = max(best_primes[L], p_finished)

            in_flight_count = max(0, in_flight_count - 1)
            pending_finished.add(p_finished)

            # advance commit
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

            # periodic print
            if time.time() - last_print > 10.0:
                print(f"commit_prime={commit_prime} dispatched_down_to={next_to_dispatch} in_flight={in_flight_count}")
                print("best_primes =", best_primes)
                elapsed = time.time() - start
                etr = int((commit_prime - RANGE_LO) / (init - commit_prime + 1) * elapsed)
                print(f"Elapsed: {elapsed:.2f}s")
                print(f"Estimated time remaining: {fmt_secs(etr)}s")

                print(f"--- Phase Breakdown ---")
                print(f"Generation: {phase_stats['gen']:.2f}s")
                print(f"Scanning:   {phase_stats['scan']:.2f}s")
                print(f"Stitching:  {phase_stats['stitch']:.2f}s")
                print(f"--- Level Breakdown ---")
                for lvl, t in prog_levels.items():
                    print(f"Level {lvl}: {t:.2f}s")
                last_print = time.time()

                print()

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
            _, gid = msg
            print(f"worker {gid} exited.")

        if next_to_dispatch <= RANGE_LO and in_flight_count == 0:
            break

    # send exit signals
    for _ in workers: prime_q.put(None)
    for w in workers: w.join(timeout=30.0)

    # final commit
    while True:
        prev_p = sympy.prevprime(commit_prime)
        if prev_p in pending_finished:
            pending_finished.remove(prev_p)
            commit_prime = int(prev_p)
            do_checkpoint(commit_prime)
        else:
            break

    do_checkpoint(commit_prime)
    print("Done. Final commit:", commit_prime)
    print("best_primes =", best_primes)
    print(f"Time elapsed: {time.time() - start}")
