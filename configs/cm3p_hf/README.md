# CM3P HF checkpoint processor config (local snapshot)

Source: <https://huggingface.co/OliBomby/CM3P> (fetched through the
`hf-mirror.com` mirror on 2026-11-05; the canonical Hugging Face endpoint was
reset from this machine, the mirror returned the same files).

This directory is the **published checkpoint's processor metadata**, not the
local training recipe in `external/cm3p/configs/train/*.yaml`.  It is committed
so T0 can compare `CM3PProcessor.from_pretrained("OliBomby/CM3P")` against a
local reference without depending on the machine that happened to have
`transformers` installed.

Files:

| file | what it controls |
|---|---|
| `processor_config.json` | `default_kwargs`: window 16 s / stride 16 s, `max_length=2000`, `truncation=longest_first` |
| `beatmap_parser/preprocessor_config.json` | parser flags: `add_snapping=false`, `add_hitsounds=true`, `add_kiai=true`, `add_mania_sv=true`, `add_timing=true` |
| `beatmap_tokenizer/tokenizer_config.json` | `add_cls_token=true`, `max_time=16000`, `time_step=10` |
| `config.json` | model architecture / hidden size (512-d projection, etc.) |

Key values (the ones T0 asserts):

    window_length_sec = 16.0
    window_stride_sec = 16.0
    max_length        = 2000
    truncation        = longest_first
    add_snapping      = false
    add_cls_token     = true
    max_time          = 16000
    time_step         = 10

The weights (`model.safetensors`) are not committed.  The checkpoint is
`OliBomby/CM3P`; the HPC copy is only a cache.
