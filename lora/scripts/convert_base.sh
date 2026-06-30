#!/usr/bin/env bash
# Export the Gemma-3-270m base WITH LoRA input slots, then bundle to .litertlm.
#
# RUN THIS ON A LINUX MACHINE WITH A PYTORCH ENV (the generative converter is
# CPU-only and officially Linux; ~16GB RAM is plenty for 270m).
#
#   python -m venv .venv && source .venv/bin/activate
#   pip install ai-edge-torch            # or: pip install litert-torch
#   pip install litert-lm-builder
#
# The --lora_ranks value MUST equal the adapters' rank r (here 64, from
# adapter_config.json). This bakes the lora_atten_{q,k,v,o}_{a,b}_prime_weight_*
# INPUT tensors into the prefill+decode signatures so adapters can be injected
# at runtime. A base exported without --lora_ranks has no injection points.
set -euo pipefail

# ---- edit these paths for your machine ----
BASE_CKPT="${BASE_CKPT:-./gemma-3-270m-it}"      # HF checkpoint dir (config.json + *.safetensors)
TOKENIZER="${TOKENIZER:-./gemma-3-270m-it/tokenizer.model}"  # SentencePiece model
OUT_DIR="${OUT_DIR:-./out}"
LORA_RANK="${LORA_RANK:-64}"
KV_MAX="${KV_MAX:-1280}"
QUANT="${QUANT:-dynamic_int8}"   # none | dynamic_int8  (int8 ~270MB, none ~1GB)
# -------------------------------------------

mkdir -p "$OUT_DIR"

echo ">> [1/2] Exporting base Gemma-3-270m to TFLite with LoRA rank ${LORA_RANK} ..."
python -m litert_torch.generative.examples.gemma3.convert_gemma3_to_tflite \
  --checkpoint_path "$BASE_CKPT" \
  --output_path "$OUT_DIR" \
  --output_name_prefix gemma3_270m \
  --model_size 270m \
  --lora_ranks "$LORA_RANK" \
  --kv_cache_max_len "$KV_MAX" \
  --quantize "$QUANT"
# (If you installed the released package, swap 'litert_torch' -> 'ai_edge_torch'.)

TFLITE_FILE="$(ls -t "$OUT_DIR"/gemma3_270m*_lora${LORA_RANK}.tflite | head -1)"
echo ">> base tflite: $TFLITE_FILE"

echo ">> [2/2] Bundling TFLite + tokenizer into .litertlm ..."
litert-lm-builder \
  system_metadata --str Authors "qualgo" \
  tflite_model --path "$TFLITE_FILE" --model_type prefill_decode \
  sp_tokenizer --path "$TOKENIZER" \
  output --path "$OUT_DIR/gemma3-270m-lora${LORA_RANK}.litertlm"

echo ">> DONE: $OUT_DIR/gemma3-270m-lora${LORA_RANK}.litertlm"
echo ">> Copy this .litertlm (and the adapter*.tflite from export_adapters.py) to the Mac."
