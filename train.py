from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

if (__name__ == "__main__"):
    # 加载模型
    max_seq_length = 8192 # 模型处理文本的最大长度
    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-1.5B",
        max_seq_length = max_seq_length,
        dtype=None, # 自动检测合适的类型
        load_in_4bit = True,
        # device_map="balanced" # 多卡训练时均衡分布模型权重，默认为sequential
    )

    # 加载数据集
    # 定义训练数据格式化字符串模板
    train_prompt_style="""请遵循指令回答用户问题。
    在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。
    ### 指令:
    你是一位精通八字算命、紫微斗数、风水、易经卦象、塔罗牌占卜、星象、面相手相和运势预测等方面的算命大师。
    请回答以下算命问题。
    ### 问题:
    {}
    ### 回答:
    <think>{}</think>
    {}
    """
    # 加载数据集
    dataset = load_dataset("data/fortune-telling", split="train")
    def formatting_data(examples):
        questions = examples["Question"]
        cots = examples["Complex_CoT"]
        responses = examples["Response"]
        texts = []
        for q, c, r in zip(questions, cots, responses):
            text = train_prompt_style.format(q, c, r) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}
    dataset = dataset.map(formatting_data, batched=True)

    # 添加 LoRA 权重
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank of the LoRA matrix
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], # Layers to apply LoRA to
        lora_alpha = 16, # LoRA alpha value
        lora_dropout = 0, # Supports any, but = 0 is optimized，防止过拟合，0 表示不drop任何参数
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # 定义 trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2, # 每个GPU上的batch size
            gradient_accumulation_steps = 4, # 梯度累积步数
            warmup_steps = 10,
            # max_steps = 200, # 最大训练步数
            num_train_epochs=3, # 训练轮数 和 max_steps 二选一
            learning_rate = 2e-4, # 学习率，默认值是 2.0e-5
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 2,
            output_dir = "outputs",
            optim = "adamw_8bit",
            seed = 3407,
        ),
    )

    # 开始训练
    train_stats = trainer.train()
    # 模型保存
    model.save_pretrained("lora/lora_model")
    tokenizer.save_pretrained("lora/lora_model")


