#!/usr/bin/env bash
set -euo pipefail

echo "VINExplainNet uses public OCT datasets."
echo "Download and place each set, then write a manifest JSON:"
echo "  { \"items\": [ { \"path\": \"frame.npy\", \"label\": 0 }, ... ] }"
echo "Kermany : https://data.mendeley.com/datasets/rscbjbr9sj/3"
echo "Duke    : https://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm"
echo "Point data.manifest in a treatment file at the manifest path to use real data."
