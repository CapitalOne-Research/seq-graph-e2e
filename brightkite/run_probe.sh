#!/usr/bin/env bash
# Run the compressed-vector probe (probe_compressed.py) that turns the §5.4 mechanism hypothesis
# into a measurement. Discovers the matched control checkpoints produced by run_controls.sh.
#
# Primary, architecture-matched comparison (same run_controls.sh grid, same MLM init):
#     frozen = *_frozenseq_matched_best.pth   (MLM-pretrained encoder, FROZEN)   -- task-agnostic z
#     joint  = *_unfrozen2stage_best.pth      (SAME MLM init, fine-tuned jointly) -- task-shaped z
# joint - frozen on the graph-free linear probe isolates "the task gradient reshapes z" with no GNN
# and no capacity change. The from-scratch end-to-end BRIDGE (no control tag) is added if present.
#
# Usage:  bash run_probe.sh [dataset=brightkite] [seed=0]
#         bash run_probe.sh brightkite 0
#         bash run_probe.sh gowalla 1
set -euo pipefail
cd "$(dirname "$0")"

DS=${1:-brightkite}
SEED=${2:-0}
CKPT_DIR="${DS}_checkpoints_seeded"
[[ -d "${CKPT_DIR}" ]] || { echo "No checkpoint dir '${CKPT_DIR}'. Run run_controls.sh first (or pass paths to probe_compressed.py directly)."; exit 1; }

# Pick the unfrozen (joint) control first, then derive the frozen control and the from-scratch
# BRIDGE from the SAME config prefix, so every arm shares architecture + context length (identical
# except the training strategy). This isolates "the task gradient reshapes z" and avoids mixing
# checkpoints trained with different --max_events / batch size (not a matched comparison).
JOINT=$(ls -1 ${CKPT_DIR}/*_seed${SEED}_unfrozen2stage_best.pth 2>/dev/null | head -1 || true)
[[ -n "${JOINT}" ]] || { echo "Missing unfrozen control (*_seed${SEED}_unfrozen2stage_best.pth) in ${CKPT_DIR}. Run run_controls.sh first."; exit 1; }
PREFIX="${JOINT%_unfrozen2stage_best.pth}"
FROZEN="${PREFIX}_frozenseq_matched_best.pth"
BRIDGE="${PREFIX}_best.pth"   # from-scratch end-to-end at the SAME config (may not exist)
[[ -f "${FROZEN}" ]] || { echo "No config-matched frozen control at ${FROZEN} (expected from run_controls.sh)."; exit 1; }

echo "[probe] dataset=${DS} seed=${SEED}  (config-matched frozen vs joint)"
echo "  frozen = ${FROZEN}"
echo "  joint  = ${JOINT}"
CKPTS=(frozen="${FROZEN}" joint="${JOINT}")
if [[ -f "${BRIDGE}" ]]; then echo "  bridge = ${BRIDGE}"; CKPTS+=(bridge="${BRIDGE}");
else echo "  bridge = (no config-matched from-scratch run at ${PREFIX}_best.pth; skipping. Pass a mismatched one to probe_compressed.py --ckpts if you want it; the probe slices its context and warns.)"; fi

python probe_compressed.py --ckpts "${CKPTS[@]}" --include_raw \
  --out "probe_${DS}_seed${SEED}.json"
echo "DONE -> probe_${DS}_seed${SEED}.json"
