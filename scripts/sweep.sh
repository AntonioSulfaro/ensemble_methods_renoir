#!/bin/bash

# List of thread counts to test for the current JSON config
THREADS=(1 2 4 8)

for T in "${THREADS[@]}"
do
    echo "---------------------------------------"
    echo "EXECUTING: Threads=$T"
    echo "---------------------------------------"

    # Run the Rust project with the --local flag for Renoir
    cargo run --profile profiling -- --local $T
done

echo "Sweep finished. Updating scalability graphs..."
py scripts/scalability_graph.py