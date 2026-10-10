# Environment used for the verified CAS audit on the Glamdring cluster (Oxford), October 2026.
# Source this before running cas_audit.py.
export PATH=/usr/local/shared/python/3.11.4/bin:/usr/local/shared/singularity/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/shared/python/3.11.4/lib
export TMPDIR=/tmp
# Giac 2.0.0 from conda-forge (micromamba environment)
export ESI_GIAC=/mnt/extraspace/hdesmond/tools/giac_env/bin/icas
# Maxima 5.47.0 from the SageMath 10.8 Singularity sandbox
export ESI_MAXIMA="/usr/local/shared/singularity/bin/singularity exec /mnt/extraspace/hdesmond/sage_sandbox /home/sage/sage/local/bin/maxima"
# FriCAS 1.3.12 Singularity image (docker://nilqed/fricas:latest); runs on comp168/comp173 only
export ESI_FRICAS="/usr/local/shared/singularity/bin/singularity exec /mnt/extraspace/hdesmond/fricas_latest.sif fricas"
# Mathematica 13.3.1 (RUBI 4.17.3.0 installed as a paclet)
export ESI_WOLFRAMSCRIPT=/usr/local/shared/mathematica/13.3/Executables/wolframscript
