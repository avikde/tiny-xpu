#!/usr/bin/env bash
# Sweep CLOCK_PERIOD_NS and tabulate area vs timing tradeoff.
#
# Usage:  ./scripts/sweep_period.sh
#
# Synthesises a **single column** (SYNTH_COLS=1) because the critical path is
# the accumulator chain down one column — all columns have identical delay.
# A full 16-column sweep at 2 ns exceeds 10 GB RAM.
# For full-array synthesis, override: cmake -DSYNTH_COLS=16 ...
#
# Results per period are saved to synth_outputs/<period>ns/ and a summary
# table is printed at the end.  Estimated full-array area ≈ column_area × 16.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJ_DIR}/build"
OUT_DIR="${PROJ_DIR}/synth_outputs"
PERIODS=(2 3 5 10 20)

echo "================================================================"
echo "  Clock Period Sweep"
echo "================================================================"
echo ""

for period in "${PERIODS[@]}"; do
    echo "--- Period=${period}ns ---"

    out_sub="${OUT_DIR}/${period}ns"
    mkdir -p "${out_sub}"

    cmake -S "${PROJ_DIR}" -B "${BUILD_DIR}" \
        -DSIM=ON -DCLOCK_PERIOD_NS="${period}" \
        -DSYNTH_ROWS=16 -DSYNTH_COLS=1 > "${out_sub}/cmake.log" 2>&1
    # Force rebuild by removing previous output — make's dependency tracking
    # can miss command-line-only changes (the -D flag value in Yosys command).
    rm -f "${OUT_DIR}/synth.json"
    make -C "${BUILD_DIR}" synth 2>&1 | tee "${out_sub}/build.log" | tail -5

    cp "${OUT_DIR}/synth_stats.txt" "${out_sub}/synth_stats.txt"
    cp "${OUT_DIR}/synth_timing.rpt" "${out_sub}/synth_timing.rpt"
    cp "${OUT_DIR}/synth.json" "${out_sub}/synth.json"

    echo ""
done

# ---- Extract & tabulate key metrics ----
echo ""
echo "================================================================"
echo "  Summary: Area vs Frequency Tradeoff (${#PERIODS[@]} periods)"
echo "  1 column (SYNTH_COLS=1) — multiply area ×16 for full 16-col array"
echo "================================================================"
printf "%-10s %-10s %-10s %-12s %-12s %-12s\n" "Period" "Freq" "Cells" "GateArea" "Area×16" "Delay"
printf "%-10s %-10s %-10s %-12s %-12s %-12s\n" "(ns)"   "(MHz)" ""      ""         ""         "(ps)"
printf "%-10s %-10s %-10s %-12s %-12s %-12s\n" "----------" "----------" "----------" "------------" "------------" "------------"

for period in "${PERIODS[@]}"; do
    stats="${OUT_DIR}/${period}ns/synth_stats.txt"
    rpt="${OUT_DIR}/${period}ns/synth_timing.rpt"

    total_cells=$(awk '/cells$/ {print $1; exit}' "${stats}")
    # The ABC summary line looks like:
    #   ABC: WireLoad = "none"  Gates = 8074 ... Area = 57935.56 ... Delay = 10483.05 ps
    abc_line=$(grep -F 'WireLoad' "${rpt}" || true)
    area=$(echo "${abc_line}" | sed -n 's/.*Area *= *\([0-9.]*\).*/\1/p')
    delay=$(echo "${abc_line}" | sed -n 's/.*Delay *= *\([0-9.]*\).*/\1/p')
    delay="${delay:-?}"
    freq_mhz=$(python3 -c "print(round(1000 / ${period}, 1))")

    if [ -n "${area}" ]; then
        area_x16=$(python3 -c "print(round(${area} * 16, 2))")
    else
        area_x16="?"
    fi

    printf "%-10s %-10s %-10s %-12s %-12s %-12s\n" \
        "${period} ns" "${freq_mhz}" "${total_cells}" "${area}" "${area_x16}" "${delay}"
done
echo ""
