#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

TESTS=(
    host_to_device_ce_ce
    host_to_device_ce_sm
    host_to_device_sm_ce
    host_to_device_sm_sm
    device_to_host_ce_ce
    device_to_host_ce_sm
    device_to_host_sm_ce
    device_to_host_sm_sm

    device_to_device_read_ce_ce
    device_to_device_read_ce_sm
    device_to_device_read_sm_ce
    device_to_device_read_sm_sm
    device_to_device_write_ce_ce
    device_to_device_write_ce_sm
    device_to_device_write_sm_ce
    device_to_device_write_sm_sm

    host_to_device_ce_device_to_device_read_ce
    host_to_device_ce_device_to_device_read_sm
    host_to_device_sm_device_to_device_read_ce
    host_to_device_sm_device_to_device_read_sm
    host_to_device_ce_device_to_device_write_ce
    host_to_device_ce_device_to_device_write_sm
    host_to_device_sm_device_to_device_write_ce
    host_to_device_sm_device_to_device_write_sm
    
    device_to_host_ce_device_to_device_read_ce
    device_to_host_ce_device_to_device_read_sm
    device_to_host_sm_device_to_device_read_ce
    device_to_host_sm_device_to_device_read_sm
    device_to_host_ce_device_to_device_write_ce
    device_to_host_ce_device_to_device_write_sm
    device_to_host_sm_device_to_device_write_ce
    device_to_host_sm_device_to_device_write_sm
)

for name in "${TESTS[@]}"; do
    echo ">>> Running: $name"
    nsys profile --trace=cuda,nvtx \
        --output="results/${name}" \
        --force-overwrite true \
        ./nvbandwidth -t "$name" --cooldown 24 -v \
        > "results/${name}.txt"
    echo "    Done: $name"
done

echo "=== All tests complete ==="
