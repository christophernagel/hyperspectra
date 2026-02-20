#!/bin/bash
# Run ISOFIT atmospheric correction in WSL
# Usage: ./run_isofit_wsl.sh <radiance.nc> <obs.nc> <output.nc>

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isofit_env

# Convert Windows paths to WSL paths
SCRIPT_DIR="/mnt/c/Users/chris/aviris_tools"
DOWNLOADS="/mnt/c/Users/chris/Downloads"

# Default test files
RDN="${1:-$DOWNLOADS/AV320230926t201618_L1B_RDN_ORT.nc}"
OBS="${2:-$DOWNLOADS/AV320230926t201618_OBS_ORT.nc}"
OUTPUT="${3:-$DOWNLOADS/AV320230926t201618_L2_REFL_isofit_wsl.nc}"

echo "============================================"
echo "ISOFIT Atmospheric Correction (WSL)"
echo "============================================"
echo "Radiance: $RDN"
echo "OBS: $OBS"
echo "Output: $OUTPUT"
echo ""

# Check files exist
if [ ! -f "$RDN" ]; then
    echo "ERROR: Radiance file not found: $RDN"
    echo "Available L1B files:"
    ls -la $DOWNLOADS/*L1B*.nc 2>/dev/null || echo "  None found"
    ls -la $DOWNLOADS/*RDN*.nc 2>/dev/null || echo "  None found"
    exit 1
fi

if [ ! -f "$OBS" ]; then
    echo "ERROR: OBS file not found: $OBS"
    echo "Available OBS files:"
    ls -la $DOWNLOADS/*OBS*.nc 2>/dev/null || echo "  None found"
    ls -la $DOWNLOADS/*ORT*.nc 2>/dev/null || echo "  None found"
    exit 1
fi

# Run ISOFIT processor
cd "$SCRIPT_DIR"
python aviris_isofit_processor.py "$RDN" "$OBS" "$OUTPUT" --cores 4

echo ""
echo "Done! Output: $OUTPUT"
