#!/usr/bin/env python3
"""Export PEFT LoRA adapters -> LiteRT-LM LoRA .tflite (attention-only).

RUN ON LINUX WITH A PYTORCH ENV (same env as convert_base.sh):
    pip install ai-edge-torch      # or: pip install litert-torch

IMPORTANT FIDELITY CAVEAT
-------------------------
LiteRT-LM dynamic LoRA injects ATTENTION ONLY (q/k/v/o). These adapters'
adapter_config.json also lists MLP modules (gate_proj/up_proj/down_proj). Those
LoRA weights CANNOT be represented on-device and are DROPPED here. The exported
adapter therefore reproduces only the attention part of the fine-tune -- good
enough to prove the swap mechanism end-to-end, but NOT a faithful deployment.
For a faithful result, retrain with target_modules=["q_proj","k_proj","v_proj","o_proj"].

scale = lora_alpha / r = 128 / 64 = 2.0  (use_rslora=false). If you change the
adapter, recompute from its adapter_config.json (rslora -> alpha/sqrt(r)).
"""

# Use 'ai_edge_torch' if you installed the released pip package instead.
from litert_torch.generative.examples.gemma3 import decoder
from litert_torch.generative.layers import lora as lora_utils

# rank/alpha read from adapter_config.json (both adapters: r=64, alpha=128).
RANK = 64
SCALE = 128 / 64  # lora_alpha / r

# Standard PEFT naming for a Gemma3ForCausalLM wrapped by PEFT.
TENSOR_NAMES = lora_utils.LoRATensorNames(
    attn_query_w_a="base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight",
    attn_query_w_b="base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight",
    attn_key_w_a="base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight",
    attn_key_w_b="base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight",
    attn_value_w_a="base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight",
    attn_value_w_b="base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight",
    attn_output_w_a="base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight",
    attn_output_w_b="base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight",
)

ADAPTERS = [
    ("../adapter_1/adapter_model.safetensors", "../out/adapter1.tflite"),
    ("../adapter_2/adapter_model.safetensors", "../out/adapter2.tflite"),
]


def main():
    config = decoder.get_decoder_config_270m()  # 18 layers, must match the base export
    for src, out in ADAPTERS:
        lora = lora_utils.LoRA.from_safetensors(
            src,
            scale=SCALE,
            config=config,
            lora_tensor_names=TENSOR_NAMES,
        )
        assert lora.get_rank() == RANK, f"rank mismatch: {lora.get_rank()} != {RANK}"
        with open(out, "wb") as f:
            f.write(lora.to_tflite())
        print(f"wrote {out}  (rank={lora.get_rank()}, scale={SCALE})")


if __name__ == "__main__":
    main()
