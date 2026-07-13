"""
desdemona_neural/splitter.py

HOT/COLD SPLIT for a dense base model (run once, after profiling):
Given the base model + a full-corpus activation profile, partition every
FFN neuron in every layer into:
  HOT  — activation freq >= threshold → stays in VRAM as FP8
  COLD — activation freq <  threshold → written to NVMe as INT4

Outputs (consumed by desdemona_neural.engine.DesdemonaEngine):
  hot_dir/layer_{L:03d}_expert_000.pt   FP8 hot-neuron tensors per layer
  hot_dir/shared.pt                      FP8 attention + FP16 embed/norms
  cold_weights.bin                       flat binary, layer-major
  cold_index.npy                         lookup: (L, 0) → (offset, n_neurons)
  split_config.json                      geometry + thresholds + counts

The dense model is stored as a single "expert 0" so the binary layout,
ColdWeightStore, hot-file naming, and predictor registry are unchanged
from the retired MoE-96 era (the per-domain expert layer was cut
2026-07-11 — see docs/BUILD_NOTES.md; this file's expert-training-era
functions went with it, recoverable from git history).

Cold weight file layout
-----------------------
For each layer L (single expert 0):
  HEADER (16 bytes):  [layer:uint16, expert:uint16, n_cold:uint32, reserved:uint64]
  DATA:               n_cold × NEURON_RECORD_BYTES, 4-KB aligned

Each NEURON_RECORD (neuron i, INT4 packed):
  neuron_id:  uint16                    (2 bytes)
  gate_row:   H/2 bytes  (INT4, H=4096) (2048 bytes)
  up_row:     H/2 bytes                 (2048 bytes)
  down_col:   H/2 bytes                 (2048 bytes)
  scales:     3 × 32 groups × FP16      (192 bytes)
  → padded to the next 64-byte boundary

cold_index.npy: shape [num_layers, 1, 2] dtype uint64 → (byte_offset, n_cold)

INT4 quantization: absmax per-neuron, group_size=128 for gate/up/down vectors.
On Blackwell (RTX 5070) this maps directly to native FP4 tensor core inputs.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


# ─── Constants ────────────────────────────────────────────────────────────────

# The binary record layout is compile-time sized for H=4096 (both Mistral-7B
# and Llama-3.1-8B geometry). split_dense() asserts the model matches.
HIDDEN_DIM        = 4096
ALIGN_BYTES       = 64        # record alignment inside cold_weights.bin
PAGE_ALIGN        = 4096      # file section alignment (NVMe page)

# Default hot budget per layer for a dense split. Each hot neuron costs
# 3 vectors × H bytes FP8 = 12 KB; 4096/layer × 32 layers ≈ 1.5 GB hot FFN,
# which plus FP8 attention (~1.3 GB) and FP16 embed/head (~0.5 GB at 32K
# vocab) fits an 8 GB card with room for KV cache and activations.
HOT_COUNT_DENSE   = 4096

_VEC_BYTES        = HIDDEN_DIM // 2                  # 2048 bytes per INT4 vector
_GROUP_SIZE          = 128
_N_GROUPS            = HIDDEN_DIM // _GROUP_SIZE     # 32 groups per vector
_SCALE_BYTES_PER_VEC = _N_GROUPS * 2                 # 64 bytes per vector (FP16)
_SCALE_BYTES_TOTAL   = _SCALE_BYTES_PER_VEC * 3      # 192 bytes

_RAW_RECORD       = 2 + _VEC_BYTES * 3 + _SCALE_BYTES_TOTAL
NEURON_RECORD_BYTES = ((_RAW_RECORD + ALIGN_BYTES - 1) // ALIGN_BYTES) * ALIGN_BYTES


# ─── INT4 quantization helpers ─────────────────────────────────────────────────

def quantize_int4(tensor: torch.Tensor, group_size: int = 128) -> tuple[bytes, torch.Tensor]:
    """
    Absmax group quantization to INT4.
    tensor: (N,) float32/float16
    Returns: (packed_bytes, scale_tensor)
    packed_bytes: N//2 bytes (two INT4 values packed per byte, little-endian)
    """
    tensor = tensor.float()
    N = tensor.numel()
    assert N % group_size == 0 or N < group_size, \
        f"tensor length {N} not divisible by group_size {group_size}"

    if N % group_size != 0:
        pad = group_size - (N % group_size)
        tensor = torch.cat([tensor, torch.zeros(pad)])

    groups  = tensor.reshape(-1, group_size)
    scales  = groups.abs().max(dim=1).values.clamp(min=1e-8) / 7.0   # INT4 range [-8,7]
    qgroups = (groups / scales.unsqueeze(1)).round().clamp(-8, 7).to(torch.int8)

    flat = qgroups.reshape(-1)[:N]
    if N % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
    lo   = flat[0::2] & 0x0F
    hi   = (flat[1::2] & 0x0F) << 4
    packed = (lo | hi).numpy().astype(np.uint8).tobytes()

    return packed, scales


def dequantize_int4(packed: bytes, scales: torch.Tensor, N: int,
                    group_size: int = 128) -> torch.Tensor:
    """Inverse of quantize_int4. Returns float32 tensor of length N."""
    arr   = np.frombuffer(packed, dtype=np.uint8)
    lo    = torch.tensor(arr & 0x0F,        dtype=torch.int8)
    hi    = torch.tensor((arr >> 4) & 0x0F, dtype=torch.int8)
    flat  = torch.stack([lo, hi], dim=1).reshape(-1)[:N].float()

    # Convert unsigned nibble to signed: values 8-15 → -8 to -1
    flat  = torch.where(flat > 7, flat - 16, flat)

    n_groups = (N + group_size - 1) // group_size
    scales_exp = scales[:n_groups].repeat_interleave(group_size)[:N]
    return flat * scales_exp


# ─── Dense hot/cold split ─────────────────────────────────────────────────────

def split_dense(
    base_model_path: str,
    profile_path: str,
    output_dir: str,
    hot_count: int = HOT_COUNT_DENSE,
    trust_remote_code: bool = False,
):
    """
    Split a dense base model's FFN neurons into hot (VRAM, FP8) and cold
    (NVMe, INT4) using per-neuron activation frequencies from profile.json
    (profiler.py records the dense model under "expert_0").

    Writes engine-ready artifacts into output_dir and returns the config dict.
    """
    out = Path(output_dir)
    hot_dir = out / "hot_weights"
    hot_dir.mkdir(parents=True, exist_ok=True)

    with open(profile_path) as f:
        profile = json.load(f)

    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    hf_cfg = model.config

    if hf_cfg.hidden_size != HIDDEN_DIM:
        raise ValueError(
            f"Base model hidden_size={hf_cfg.hidden_size} but the cold-record "
            f"binary layout is sized for {HIDDEN_DIM}. Pick a 4096-hidden model "
            f"or resize the record constants."
        )

    num_layers = hf_cfg.num_hidden_layers
    inter_dim  = hf_cfg.intermediate_size

    print(f"Dense hot/cold split | hot_count={hot_count}/{inter_dim} neurons/layer "
          f"| {num_layers} layers | act={hf_cfg.hidden_act}")
    t0 = time.time()

    # ── Shared weights (always hot) ────────────────────────────────────────
    print("Extracting shared attention weights...")
    shared: dict[str, torch.Tensor] = {}
    for li, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = getattr(layer.self_attn, name).weight.data
            shared[f"layer_{li:03d}.{name}"] = w.to(torch.float8_e4m3fn)
        shared[f"layer_{li:03d}.input_layernorm"]          = layer.input_layernorm.weight.data.half()
        shared[f"layer_{li:03d}.post_attention_layernorm"] = layer.post_attention_layernorm.weight.data.half()
    shared["embed_tokens"] = model.model.embed_tokens.weight.data.half()
    shared["lm_head"]      = model.lm_head.weight.data.half()
    shared["final_norm"]   = model.model.norm.weight.data.half()
    torch.save(shared, str(hot_dir / "shared.pt"))
    print(f"  Shared weights saved ({sum(v.numel() for v in shared.values())/1e9:.2f}B params)")

    # ── Per-layer hot/cold partition ───────────────────────────────────────
    cold_path  = out / "cold_weights.bin"
    cold_index = np.zeros((num_layers, 1, 2), dtype=np.uint64)

    total_hot = total_cold = 0
    byte_cursor = 0

    with open(cold_path, "wb") as cold_f:
        for li in range(num_layers):
            mlp  = model.model.layers[li].mlp
            gate = mlp.gate_proj.weight.data.float()   # (I, H)
            up   = mlp.up_proj.weight.data.float()     # (I, H)
            down = mlp.down_proj.weight.data.float()   # (H, I)
            I    = gate.shape[0]

            freq_list = profile["layers"].get(str(li), {}).get("expert_0", [])
            if freq_list and len(freq_list) == I:
                freq = torch.tensor(freq_list)
                top_indices = torch.topk(freq, min(hot_count, I)).indices
                hot_mask = torch.zeros(I, dtype=torch.bool)
                hot_mask[top_indices] = True
            else:
                raise ValueError(
                    f"profile has no usable expert_0 frequencies for layer {li} "
                    f"(got {len(freq_list)}, need {I}) — re-run profiler.py on "
                    f"this exact base model first."
                )

            hot_indices  = hot_mask.nonzero(as_tuple=True)[0]
            cold_indices = (~hot_mask).nonzero(as_tuple=True)[0]
            n_cold = len(cold_indices)
            total_hot  += len(hot_indices)
            total_cold += n_cold

            # Hot neurons → FP8 file (expert index frozen at 0)
            torch.save({
                "neuron_ids": hot_indices,
                "gate":       gate[hot_indices].to(torch.float8_e4m3fn),
                "up":         up[hot_indices].to(torch.float8_e4m3fn),
                "down":       down[:, hot_indices].to(torch.float8_e4m3fn),
            }, str(hot_dir / f"layer_{li:03d}_expert_000.pt"))

            # Cold neurons → flat binary
            section_start = byte_cursor
            header = struct.pack("<HHIq", li, 0, n_cold, 0)
            cold_f.write(header)
            byte_cursor += len(header)

            for nid in cold_indices.tolist():
                gate_packed, gate_scales = quantize_int4(gate[nid])
                up_packed,   up_scales   = quantize_int4(up[nid])
                down_packed, down_scales = quantize_int4(down[:, nid])

                record = struct.pack("<H", nid)
                record += gate_packed + up_packed + down_packed
                record += gate_scales.to(torch.float16).numpy().tobytes()
                record += up_scales.to(torch.float16).numpy().tobytes()
                record += down_scales.to(torch.float16).numpy().tobytes()

                pad = (ALIGN_BYTES - len(record) % ALIGN_BYTES) % ALIGN_BYTES
                record += b"\x00" * pad
                cold_f.write(record)
                byte_cursor += len(record)

            remainder = byte_cursor % PAGE_ALIGN
            if remainder:
                pad = PAGE_ALIGN - remainder
                cold_f.write(b"\x00" * pad)
                byte_cursor += pad

            cold_index[li, 0] = (section_start, n_cold)

            if (li + 1) % 8 == 0:
                print(f"  Layer {li+1:2d}/{num_layers}  "
                      f"({time.time()-t0:.1f}s, {byte_cursor/1e9:.2f} GB written)")

    np.save(str(out / "cold_index.npy"), cold_index)

    # Engine + ColdWeightStore geometry, sourced from the model's own config
    config = {
        "base_model":            base_model_path,
        "hidden_dim":            hf_cfg.hidden_size,
        "intermediate_dim":      inter_dim,
        "expert_inter_dim":      inter_dim,      # legacy key, same value
        "num_experts":           1,
        "num_layers":            num_layers,
        "vocab_size":            hf_cfg.vocab_size,
        "num_attention_heads":   hf_cfg.num_attention_heads,
        "num_key_value_heads":   hf_cfg.num_key_value_heads,
        "rope_theta":            getattr(hf_cfg, "rope_theta", 10000.0),
        "rms_norm_eps":          hf_cfg.rms_norm_eps,
        "hidden_act":            hf_cfg.hidden_act,
        # dReLU family (TurboSparse/Bamboo) applies the activation to BOTH
        # gate and up branches — act(g)⊙act(u) — unlike standard gated FFNs
        # (act(g)⊙u). Verified against modeling_bamboo.py's BambooMLP.
        "act_on_both":           getattr(hf_cfg, "model_type", "") == "bamboo",
        "max_position_embeddings": hf_cfg.max_position_embeddings,
        "hot_count":             hot_count,
        "total_hot_neurons":     int(total_hot),
        "total_cold_neurons":    int(total_cold),
        "hot_fraction":          total_hot / max(total_hot + total_cold, 1),
        "cold_file_bytes":       byte_cursor,
        "neuron_record_bytes":   NEURON_RECORD_BYTES,
        "page_align":            PAGE_ALIGN,
    }
    with open(out / "split_config.json", "w") as f:
        json.dump(config, f, indent=2)

    del model

    print(f"\nSplit complete in {time.time()-t0:.1f}s")
    print(f"  Hot  neurons: {total_hot:,}  ({config['hot_fraction']*100:.1f}%)")
    print(f"  Cold neurons: {total_cold:,}  ({(1-config['hot_fraction'])*100:.1f}%)")
    print(f"  Cold file:    {byte_cursor/1e9:.2f} GB → {cold_path}")
    print(f"  Hot dir:      {hot_dir}")

    return config


# ─── Cold weight reader (used by engine) ──────────────────────────────────────

class ColdWeightStore:
    """
    Memory-maps cold_weights.bin and provides fast neuron lookups
    using the cold_index.npy offset table.

    Used by the inference engine to fetch specific cold neurons
    given a (layer, expert, neuron_id_list) request (expert is 0
    for dense splits).
    """

    def __init__(self, cold_bin: str, cold_index: str, config: dict):
        import mmap
        self.config = config
        self.index  = np.load(cold_index)    # shape (L, E, 2)

        self._f  = open(cold_bin, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        self.H = config["hidden_dim"]
        self.I = config.get("intermediate_dim", config.get("expert_inter_dim"))

        # Per-(layer, expert) neuron_id → byte-offset maps, built lazily on
        # first access so repeated reads skip the section header scan.
        self._offset_cache: dict[tuple[int, int], dict[int, int]] = {}

    def _section_offsets(self, layer_idx: int, expert_idx: int) -> dict[int, int] | None:
        key = (layer_idx, expert_idx)
        cached = self._offset_cache.get(key)
        if cached is not None:
            return cached

        section_offset, n_cold = self.index[layer_idx, expert_idx]
        if n_cold == 0:
            return None

        header_size = 16   # u16 + u16 + u32 + u64
        cursor      = int(section_offset) + header_size

        id_to_offset: dict[int, int] = {}
        for _ in range(int(n_cold)):
            nid = struct.unpack_from("<H", self._mm, cursor)[0]
            id_to_offset[nid] = cursor
            cursor += NEURON_RECORD_BYTES

        self._offset_cache[key] = id_to_offset
        return id_to_offset

    def read_neurons(
        self,
        layer_idx:  int,
        expert_idx: int,
        neuron_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Fetch specific cold neurons for (layer, expert).
        Returns (gate, up, down) float32 tensors shaped (N, H), (N, H), (H, N).
        Returns None if the section is empty or no requested id exists.
        """
        id_to_offset = self._section_offsets(layer_idx, expert_idx)
        if id_to_offset is None:
            return None

        results_gate, results_up, results_down = [], [], []
        for nid in neuron_ids:
            if nid not in id_to_offset:
                continue
            off = id_to_offset[nid] + 2   # skip neuron_id u16

            gate_b = bytes(self._mm[off               : off + _VEC_BYTES])
            up_b   = bytes(self._mm[off + _VEC_BYTES  : off + _VEC_BYTES*2])
            down_b = bytes(self._mm[off + _VEC_BYTES*2: off + _VEC_BYTES*3])

            scale_off = off + _VEC_BYTES * 3
            gate_scales = torch.frombuffer(
                bytes(self._mm[scale_off                          : scale_off + _SCALE_BYTES_PER_VEC]),
                dtype=torch.float16,
            ).float()
            up_scales = torch.frombuffer(
                bytes(self._mm[scale_off + _SCALE_BYTES_PER_VEC  : scale_off + _SCALE_BYTES_PER_VEC*2]),
                dtype=torch.float16,
            ).float()
            down_scales = torch.frombuffer(
                bytes(self._mm[scale_off + _SCALE_BYTES_PER_VEC*2: scale_off + _SCALE_BYTES_TOTAL]),
                dtype=torch.float16,
            ).float()

            results_gate.append(dequantize_int4(gate_b, gate_scales, self.H))
            results_up.append(dequantize_int4(up_b,   up_scales,   self.H))
            results_down.append(dequantize_int4(down_b, down_scales, self.H))

        if not results_gate:
            return None

        gate  = torch.stack(results_gate)          # (N, H)
        up    = torch.stack(results_up)            # (N, H)
        down  = torch.stack(results_down, dim=1)   # (H, N)
        return gate, up, down

    def close(self):
        self._mm.close()
        self._f.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    sd = sub.add_parser("split-dense",
                        help="Split a dense base model into hot/cold weights")
    sd.add_argument("--base",       required=True, help="HF id or local model path")
    sd.add_argument("--profile",    required=True, help="profile.json from profiler.py")
    sd.add_argument("--output",     default="split/")
    sd.add_argument("--hot-count",  type=int, default=HOT_COUNT_DENSE,
                    help=f"Top-N neurons per layer kept in VRAM as FP8 "
                         f"(default {HOT_COUNT_DENSE})")
    sd.add_argument("--trust-remote-code", action="store_true",
                    help="Pass through to from_pretrained (some sparsified "
                         "checkpoints ship custom modeling code)")

    args = parser.parse_args()

    if args.cmd == "split-dense":
        split_dense(args.base, args.profile, args.output,
                    args.hot_count, args.trust_remote_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
