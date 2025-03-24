#!/bin/bash

# Set the job ID to check
JOB_ID=29664614

echo "Checking running job $JOB_ID..."

# Get job info
echo "=== JOB INFO ==="
squeue -j $JOB_ID -l

# Check if job has created temporary files
echo -e "\n=== TEMPORARY OUTPUT ==="
find /scratch/midway3/$USER -type f -name "*$JOB_ID*" 2>/dev/null || echo "No temporary files found"

# Check if job logs exist
echo -e "\n=== JOB LOGS ==="
ls -la slurm-$JOB_ID.out 2>/dev/null || echo "No stdout log file found"
ls -la slurm-$JOB_ID.err 2>/dev/null || echo "No stderr log file found"

# If log files exist, show content
if [ -f "slurm-$JOB_ID.out" ]; then
  echo -e "\n=== LOG CONTENT (TAIL) ==="
  tail -n 50 slurm-$JOB_ID.out
fi

# Check for any new files created recently
echo -e "\n=== RECENT FILES IN WORKING DIRECTORY ==="
find . -type f -mmin -60 -not -path "*/\.*" | sort

echo "Check complete." 