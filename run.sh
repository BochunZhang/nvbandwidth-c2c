#!/usr/bin/env bash
# run.sh – nvbandwidth test runner
# Outputs: results/standard/{CATEGORY}/{test}.json
#          results/custom/{CATEGORY}/{test}.json
set -euo pipefail

# ── helper ────────────────────────────────────────────────────────────────────
# run_test <output_dir> <testcase_name>
run_test() {
    local dir="$1"
    local name="$2"
    mkdir -p "$dir"
    echo ">>> Running: $name"
    nsys profile --trace=cuda,nvtx \
        --output="$dir/${name}" \
        --force-overwrite true \
        ./nvbandwidth -t "$name" --cooldown 24 --json \
        > "$dir/${name}.json"
    echo "    Done: $name"
}

# run_group <output_dir> <test1> <test2> ...
run_group() {
    local dir="$1"; shift
    for name in "$@"; do
        run_test "$dir" "$name"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD TESTS  (built-in nvbandwidth, single-stream measurement)
# → results/standard/{CATEGORY}/
# ─────────────────────────────────────────────────────────────────────────────

STD_H2D=(
    host_to_device_memcpy_ce
    host_to_device_memcpy_sm
    host_to_all_memcpy_ce
    host_to_all_memcpy_sm
    host_to_all_bidirectional_memcpy_ce
    host_to_all_bidirectional_memcpy_sm
)

STD_D2H=(
    device_to_host_memcpy_ce
    device_to_host_memcpy_sm
    all_to_host_memcpy_ce
    all_to_host_memcpy_sm
    all_to_host_bidirectional_memcpy_ce
    all_to_host_bidirectional_memcpy_sm
)

STD_D2D_R=(
    device_to_device_memcpy_read_ce
    device_to_device_memcpy_read_sm
    device_to_device_bidirectional_memcpy_read_ce
    device_to_device_bidirectional_memcpy_read_sm
    all_to_one_read_ce
    all_to_one_read_sm
    one_to_all_read_ce
    one_to_all_read_sm
)

STD_D2D_W=(
    device_to_device_memcpy_write_ce
    device_to_device_memcpy_write_sm
    device_to_device_bidirectional_memcpy_write_ce
    device_to_device_bidirectional_memcpy_write_sm
    all_to_one_write_ce
    all_to_one_write_sm
    one_to_all_write_ce
    one_to_all_write_sm
)

STD_HD_BIDIR=(
    host_to_device_bidirectional_memcpy_ce
    host_to_device_bidirectional_memcpy_sm
    device_to_host_bidirectional_memcpy_ce
    device_to_host_bidirectional_memcpy_sm
)

STD_MISC=(
    device_local_copy
    host_device_latency_sm
    device_to_device_latency_sm
)

echo "=== Standard Tests ==="
run_group "results/standard/H2D"       "${STD_H2D[@]}"
run_group "results/standard/D2H"       "${STD_D2H[@]}"
run_group "results/standard/D2D_R"     "${STD_D2D_R[@]}"
run_group "results/standard/D2D_W"     "${STD_D2D_W[@]}"
run_group "results/standard/HD_BIDIR"  "${STD_HD_BIDIR[@]}"
run_group "results/standard/MISC"      "${STD_MISC[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TESTS  (concurrent/parallel dual-stream tests)
# → results/custom/{CATEGORY}/
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_HD_BIDIR=(
    host_to_device_bidirectional_memcpy_ce_ce
    host_to_device_bidirectional_memcpy_ce_sm
    host_to_device_bidirectional_memcpy_sm_ce
    host_to_device_bidirectional_memcpy_sm_sm
)

CUSTOM_H2D=(
    host_to_device_ce_ce
    host_to_device_ce_sm
    host_to_device_sm_ce
    host_to_device_sm_sm
)

CUSTOM_D2H=(
    device_to_host_ce_ce
    device_to_host_ce_sm
    device_to_host_sm_ce
    device_to_host_sm_sm
)

CUSTOM_D2D_R=(
    device_to_device_read_ce_ce
    device_to_device_read_ce_sm
    device_to_device_read_sm_ce
    device_to_device_read_sm_sm
)

CUSTOM_D2D_W=(
    device_to_device_write_ce_ce
    device_to_device_write_ce_sm
    device_to_device_write_sm_ce
    device_to_device_write_sm_sm
)

CUSTOM_H2D_D2D_R=(
    host_to_device_ce_device_to_device_read_ce
    host_to_device_ce_device_to_device_read_sm
    host_to_device_sm_device_to_device_read_ce
    host_to_device_sm_device_to_device_read_sm
)

CUSTOM_H2D_D2D_W=(
    host_to_device_ce_device_to_device_write_ce
    host_to_device_ce_device_to_device_write_sm
    host_to_device_sm_device_to_device_write_ce
    host_to_device_sm_device_to_device_write_sm
)

CUSTOM_D2H_D2D_R=(
    device_to_host_ce_device_to_device_read_ce
    device_to_host_ce_device_to_device_read_sm
    device_to_host_sm_device_to_device_read_ce
    device_to_host_sm_device_to_device_read_sm
)

CUSTOM_D2H_D2D_W=(
    device_to_host_ce_device_to_device_write_ce
    device_to_host_ce_device_to_device_write_sm
    device_to_host_sm_device_to_device_write_ce
    device_to_host_sm_device_to_device_write_sm
)

CUSTOM_CONCURRENT=(
    concurrent_ce
)

echo "=== Custom Tests ==="
run_group "results/custom/HD_BIDIR"    "${CUSTOM_HD_BIDIR[@]}"
run_group "results/custom/H2D"         "${CUSTOM_H2D[@]}"
run_group "results/custom/D2H"         "${CUSTOM_D2H[@]}"
run_group "results/custom/D2D_R"       "${CUSTOM_D2D_R[@]}"
run_group "results/custom/D2D_W"       "${CUSTOM_D2D_W[@]}"
run_group "results/custom/H2D_D2D_R"   "${CUSTOM_H2D_D2D_R[@]}"
run_group "results/custom/H2D_D2D_W"   "${CUSTOM_H2D_D2D_W[@]}"
run_group "results/custom/D2H_D2D_R"   "${CUSTOM_D2H_D2D_R[@]}"
run_group "results/custom/D2H_D2D_W"   "${CUSTOM_D2H_D2D_W[@]}"
run_group "results/custom/CONCURRENT"  "${CUSTOM_CONCURRENT[@]}"

echo "=== All tests complete ==="
echo "Run: python3 analyze_results.py  to generate Excel reports"
