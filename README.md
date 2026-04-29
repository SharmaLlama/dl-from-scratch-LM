# dl-from-scratch-LM
A repo for recreating interesting papers in language modeling tasks to further my own understanding of SOTA LLMs.



## Things changed as models scaled up: 

1. Small LLM
- mixed precision 
- gradient accumulation
- expandable segments to true


2. Medium LLM
- changed to bf16 instead of fp16
- compile 
- gradient checkpointing
- fused AdamW
- gating capture fix
- weight tying
- no per-step sync
- DDP no_sync()
- TF.set_float32_matmul_precision("high")
