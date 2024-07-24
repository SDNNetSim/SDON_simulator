#!/bin/bash

# This script calculates the total memory usage of processes matching a given name and counts the number of instances
# of a script running. It is intended to be used on Unix-like systems.

# Usage: ./script.sh <script_path> <process_name>
# Example: ./script.sh /path/to/script.sh process_name

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <script_path> <process_name>"
  exit 1
fi

# Assign arguments to variables for better readability
script_path="$1"
script_name=$(basename "$script_path")
process_name="$2"

# Check if the provided script path is valid
if [ ! -f "$script_path" ]; then
  echo "Error: $script_path is not a valid file path."
  exit 1
fi

# Find the PIDs of processes matching the process name
pids=$(pgrep -x "$process_name")
if [ -z "$pids" ]; then
  echo "No processes found for $process_name."
  exit 1
fi

# Calculate the total memory usage of the processes
ps u -p "$pids" | awk '{sum += $6} END {print "Total memory used is: " sum / 1024 " MB" " or " sum / 1024 / 1024 " GB"}'

# Count the number of instances of the script running
num_proc=$(pgrep -xc "$script_name")
echo "Total number of processes used by $script_name is: $num_proc"
