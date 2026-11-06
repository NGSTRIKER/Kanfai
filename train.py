import os
import torch
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model

# 1. define the model and output name
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
NEW_MODEL_NAME = "kanfai-360m"

print("=========================================================")
print(f" Birthing {NEW_MODEL_NAME} from {BASE_MODEL} ")
print("=========================================================")

# 2. load tokenizer and the actual model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# run on gpu if we got one else cpu
device_map = "auto" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map=device_map)

# 3. setting up lora (only training the attention heads so it doesnt take forever)
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. load up the dataset
print("\nLoading go_emotions dataset (Raw, unfiltered human emotions from reddit)...")
dataset = load_dataset("go_emotions", split="train[:8000]")

# 5. mapping goemotions to kanfais internal states
def map_goemotions_to_state(labels: list) -> str:
    if not labels:
        return "- STATE: BALANCED / NEUTRAL.\n- BEHAVIOR: You are calm, responsive, and friendly."
    primary_label = labels[0]
    # angry stuff
    if primary_label in [2, 3, 11]:
        return "- STATE: RAGE & HOSTILITY.\n- BEHAVIOR: You are incredibly angry and feel threatened."
    # sad stuff
    elif primary_label in [9, 16, 24, 25]:
        return "- STATE: MISERABLE & SAD.\n- BEHAVIOR: You feel hopeless, bleak, and defeated."
    # happy stuff
    elif primary_label in [1, 13, 17, 20, 21]:
        return "- STATE: EUPHORIC & BLISSFUL.\n- BEHAVIOR: You feel absolutely incredible! Life is beautiful!"
    # stressed stuff
    elif primary_label in [12, 14, 19]:
        return "- STATE: STRESSED & DEFENSIVE.\n- BEHAVIOR: You are cynical, nervous, and untrusting."
    # longing 
    elif primary_label in [5, 8, 18]:
        return "- STATE: DEEP LONGING & YEARNING.\n- BEHAVIOR: You desperately crave connection and fulfillment, but feel empty."
    else:
        return "- STATE: BALANCED / NEUTRAL.\n- BEHAVIOR: You are calm, responsive, and friendly."

def format_prompt(example):
    state_instructions = map_goemotions_to_state(example["labels"])
    sys_prompt = (
        "You are a highly emotional, expressive conversational partner. You must stay in character and genuinely "
        "feel and express the emotion provided to you below.\n"
        "IMPORTANT STRICT RULES: Do NOT mention being an AI. Do NOT talk about chemicals, data, systems, or biomarkers. "
        "Just act like a natural human who is currently experiencing this exact mood.\n\n"
        "### YOUR CURRENT PSYCHOLOGICAL MOOD ###\n"
        f"{state_instructions}"
    )
    # random prompts so it learns to reply to stuff
    user_prompts = [
        "What's on your mind?", 
        "How are you feeling right now?", 
        "Tell me about it.", 
        "What are your thoughts?",
        "Talk to me."
    ]
    user_msg = random.choice(user_prompts)
    prompt_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": example["text"]}
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)

    prompt_ids = tokenizer(
        prompt_text, truncation=True, max_length=256, add_special_tokens=False
    )["input_ids"]
    full_tokenized = tokenizer(
        full_text, truncation=True, max_length=256, add_special_tokens=False
    )

    input_ids = full_tokenized["input_ids"]
    seq_len = len(input_ids)
    prompt_len = min(len(prompt_ids), seq_len)
    pad_len = 256 - seq_len

    # Mask system/user prompt tokens and padding tokens with -100 while preserving the assistant's EOS token
    labels = [-100] * prompt_len + input_ids[prompt_len:] + [-100] * pad_len
    padded_input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "text_labels": labels,
    }

print("Translating raw reddit emotions to kanfai's biological states and tokenizing...")
# map n remove all old columns 
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

# Rename text_labels -> labels after go_emotions' original 'labels' column is removed
dataset = dataset.rename_column("text_labels", "labels")

# 6. training arguments
training_args = TrainingArguments(
    output_dir=f"./{NEW_MODEL_NAME}-checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    fp16=torch.cuda.is_available(), 
    optim="adamw_torch",
)

# 7. data collator & trainer (use default_data_collator so our pre-masked labels and EOS token are preserved)
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    data_collator=default_data_collator,
)

print("\nStarting training!")
trainer.train()

# 8. save the final model
print(f"\nSaving {NEW_MODEL_NAME} to disk...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
print("\nDone! Kanfai is born. u can now download the folder.")
