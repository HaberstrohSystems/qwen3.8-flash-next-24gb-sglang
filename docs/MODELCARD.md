# Model card

The model card lives with the weights on the Hugging Face Hub:
https://huggingface.co/HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang

[WRITEUP.md](WRITEUP.md) sections 1-2 describe the AutoRound run; afterwards the 85 dense tensors
(`lm_head`, the QSA `q/k/v/o_proj`, the GDN `out_proj`) were re-packed from bf16 to INT8 g128 by
[`../scripts/requant_int8.py`](../scripts/requant_int8.py) (CAMPAIGN.md:298-300). The Hub card's
precision table is the shipped state. The serving configuration is
[`../scripts/serve.sh`](../scripts/serve.sh).
