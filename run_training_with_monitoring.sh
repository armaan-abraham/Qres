#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p qres/data

# Start GPU memory monitoring in the background
(
  while true; do
    # Append timestamp
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> qres/data/gpu_mem.log
    # Log GPU memory usage
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv,noheader,nounits >> qres/data/gpu_mem.log
    sleep 2
  done
) &

# Record the PID of the GPU monitoring process
GPUMONITOR_PID=$!

# Start system memory monitoring in the background
(
  while true; do
    # Append timestamp
    echo "$(date '+%Y-%m-%d %H:%M:%S')" >> qres/data/sys_mem.log
    # Log system memory usage
    free -h >> qres/data/sys_mem.log
    sleep 2
  done
) &

# Record the PID of the system memory monitoring process
SYSMONITOR_PID=$!

# Run the training script, redirecting stdout and stderr to train.log
python qres/train.py > qres/data/train.log 2>&1

# After the training script finishes, kill the monitoring processes
kill $GPUMONITOR_PID
kill $SYSMONITOR_PID

echo "Training completed. Monitoring processes terminated." 