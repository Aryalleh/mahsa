#!/usr/bin/env bash
# Launch Mahsa with the GPU library path set.
#
# On WSL, llama.cpp must load the driver's libcuda from /usr/lib/wsl/lib; a
# conflicting libcuda from the CUDA toolkit otherwise shadows it and llama.cpp
# silently falls back to CPU (very slow). Running through this script guarantees
# the GPU is used. Just run:  bash run.sh
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
exec python3 main.py "$@"
