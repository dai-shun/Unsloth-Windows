import os
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora/lora_model", # 加载训练后的LoRA模型
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
alpaca_prompt = """请遵循指令回答用户问题。在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。
### 指令:
{}
### 问题:
{}
### 回答:
{}"""

inputs = tokenizer(
[
    alpaca_prompt.format(
        "你是一位精通八字算命、紫微斗数、风水、易经卦象、塔罗牌占卜、星象、面相手相和运势预测等方面的算命大师。请回答以下算命问题。",
        "国历2000年01月01日生，今年是2025年，了解未来五年的运势",
        "",
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 12800)