#### ❓Question #1:

What exactly is happening in the double quantization step?

> NOTE: You can use the paper provided to find the answer!

So the simple answer from papers is: "Double Quantization, a method that quantizes the quantization constants"

At the start, we have a single quantization, where we convert heavy weighted 32-bit floats to much smaller 8-bit integers. 
This is the main goal and purpose of single quantization.

But we go deeper, and we're applying double quantization. When we did the first quantization, 
it also saved some additional, let's call it metadata of quantization. This metadata is also floats, 
and it took additional storage space in our memory. So the idea is to quantize the metadata of first quantization.

#### ❓Question #2:

![image](https://i.imgur.com/N8y2crZ.png)

Label the image with the appropriate layer from `meta-llama/Llama-3.1-8B-Instruct`'s architecture.

- EXAMPLE - Layer Norm:
  - `(input_layernorm): LlamaRMSNorm()`
  - `(post_attention_layernorm): LlamaRMSNorm()`
  - `(norm): LlamaRMSNorm()`
- Feed Forward:
  - `(mlp): LlamaMLP` 
- Masked Multi Self-Attention:
  - `(self_attn): LlamaAttention`
- Text & Position Embed:
  - `(embed_tokens): Embedding(128256, 4096, padding_idx=128004)`
- Text Prediction:
  - `(lm_head): Linear(in_features=4096, out_features=128256, bias=False)` 

#### ❓Question #3:

What modules (or groupings of layers) did we apply LoRA too - and how can we tell from the model summary?

The main indicator that shows did we apply LoRA for me was `lora.Linear` instead of default `Linear`. 
So this replacement signals that we use LoRA for optimization.

Here is the final list of layers that I detect in `print(model)` output:
- LlamaAttention (q_proj, k_proj, v_proj, o_proj)
- LlamaMLP (gate_proj, up_proj, down_proj)

#### ❓Question #4:

Describe what the following parameters are doing:

- `warmup_ratio`: total training steps used for a linear warmup.
- `learning_rate`: learning rate for AdamW optimizer. This optimizer implements Adam algorithm with weight decay.
- `lr_scheduler_type`: type of scheduler to use. There are some options: linear, cosine, cosine_with_restarts, polinomial and more.

> NOTE: Feel free to consult the [documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) or other resources!
