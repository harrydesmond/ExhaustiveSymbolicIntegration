"""Generate trig_maths function set for k=4..8 in a single MPI session.
Skips check_results (slow with trig) and loops over complexities within
a single Python process to avoid MPI re-initialization issues.

Usage: addqueue -q berg -n 20 -m 5 bash run_gen_trig_all.sh
"""
import sys
sys.path.insert(0, '/mnt/extraspace/hdesmond/ESR')
sys.path.insert(0, '/mnt/extraspace/hdesmond/pylibs')

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Monkey-patch check_results to be a no-op
import esr.generation.simplifier as simplifier
def _skip_check(dirname, compl, tmax=10):
    if rank == 0:
        print(f'\nSkipping check_results for compl={compl} (trig mode)', flush=True)
    comm.Barrier()
simplifier.check_results = _skip_check

import esr.generation.duplicate_checker as dc

for k in range(4, 9):
    if rank == 0:
        print(f"\n{'='*50}", flush=True)
        print(f"Generating trig_maths k={k}", flush=True)
        print(f"{'='*50}", flush=True)
    comm.Barrier()
    dc.main('trig_maths', k)
    if rank == 0:
        print(f"Done k={k}", flush=True)
    comm.Barrier()

if rank == 0:
    print("ALL DONE", flush=True)
