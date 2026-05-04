"""Generate trig_maths function set for a given complexity using MPI.
Usage: addqueue -n 10 -m 5 /usr/local/shared/python/3.11.4/bin/python3 gen_trig_mpi.py <k>
"""
import sys
sys.path.insert(0, '/mnt/extraspace/hdesmond/ESR')
sys.path.insert(0, '/mnt/extraspace/hdesmond/pylibs')

import esr.generation.duplicate_checker as dc

k = int(sys.argv[1])
print(f"Generating trig_maths k={k}", flush=True)
dc.main('trig_maths', k)
print(f"Done k={k}", flush=True)
