# BOOOM-MATLAB-GPU
For Das Lab internal use only!

# Work-flow
1) Open 'BOOOM_gpu_single_precision.m', uses GPU-parallel.
2) Cluster: At 'athena.hprc.vcu.edu' use NVIDIA H100 GPU cluster in MATLAB. Check time to converge.
3) Time complexity is proportional to Q^2 or similar. Increasing P or M should not affect computation time that much.
4) Run for P=1000, 10000, Q = 500, 1000. See how much time it is taking. M > = P > = Q.
5) Final goal: Run for P x Q = 10000 x 1000, N = 10000/100000 .
