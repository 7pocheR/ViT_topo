#!/bin/bash

# Set the job ID to search for
JOB_ID=29665112

echo "Searching for output files for job $JOB_ID..."

# Look for Slurm output files with this job ID
echo "=== SLURM OUTPUT FILES ==="
find ~ -name "slurm-$JOB_ID*.out" -o -name "*vit_topo*$JOB_ID*.out" | while read file; do
  echo "Found file: $file"
  echo "File size: $(du -h "$file" | cut -f1)"
  echo "File content (first 20 lines):"
  head -n 20 "$file"
  echo "File content (last 20 lines):"
  tail -n 20 "$file"
  echo "=============================="
done

# Look for error files
echo "=== SLURM ERROR FILES ==="
find ~ -name "slurm-$JOB_ID*.err" -o -name "*vit_topo*$JOB_ID*.err" | while read file; do
  echo "Found file: $file"
  echo "File size: $(du -h "$file" | cut -f1)"
  echo "File content:"
  cat "$file"
  echo "=============================="
done

# Look for result files created around the time the job ran
JOB_DATE="2025-03-24"
echo "=== CHECKING FOR RESULTS ==="
echo "Files modified on $JOB_DATE in home directory:"
find ~ -type f -newermt "$JOB_DATE 00:00:00" ! -newermt "$JOB_DATE 23:59:59" -name "*.json" -o -name "*.png" | sort

# Check recently created files in home directory and subdirectories
echo "=== RECENT FILES ==="
find ~ -type f -mtime -2 -name "*.json" -o -name "*.png" | sort

echo "Search complete." 