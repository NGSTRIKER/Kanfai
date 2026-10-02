import os
import sys
import threading
from typing import Dict

import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

try:
    import emotion_engine
except ImportError:
    print("Pls compile the emotion_engine first. Run in emotion_engine directory:")
    print("maturin develop --release")
    sys.exit(1)

# intialize the rust emotion state (this is the biological brain)
state = emotion_engine.EmotionState()

# toggle this to False if u want kanfai to talk normal instead of spittin out reddit comments lol
USE_REDDIT_BRAIN = False

# load the model (loading on cpu cuz i got 8gb ram only)
if USE_REDDIT_BRAIN and os.path.exists("kanfai-360m"):
    model_id = "kanfai-360m"
    print(f"Loading REDDIT KANFAI ({model_id}) on CPU...")
else:
    model_id = "thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored"
    print(f"Loading UPGRADED KANFAI ({model_id}) on CPU... (give it a min to download!)")

tokenizer = AutoTokenizer.from_pretrained(model_id)
# using bfloat16 so it doesnt crash my pc
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cpu")

# setting up the matplotlib graphs for the chemicals
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.canvas.manager.set_window_title("20-Chemical Visualization Matrix")
fig.suptitle("Real-time Neurochemical Status", fontsize=16)

# groupin the chemicals together for the 4 plots
groups = {
    "Excitatory & Drive": ["dopamine", "norepinephrine", "glutamate", "histamine", "adrenaline"],
    "Inhibitory & Mood": ["serotonin", "gaba", "glycine", "prolactin", "dhea"],
    "Stress & Survival": ["cortisol", "oxytocin", "vasopressin", "endorphin", "acetylcholine"],
    "Metabolic & Sleep": ["adenosine", "melatonin", "ghrelin", "leptin", "insulin"]
}

history = {k: {chem: [] for chem in v} for k, v in groups.items()}
time_steps = []

def update_plot(state_dict: Dict[str, float], step: int):
    time_steps.append(step)
    
    for ax in axs.flat:
        ax.clear()
        
    for idx, (title, chemicals) in enumerate(groups.items()):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for chem in chemicals:
            history[title][chem].append(state_dict[chem])
            ax.plot(time_steps, history[title][chem], label=chem.capitalize(), marker='o', markersize=4)
        
        ax.legend(loc="upper left", fontsize="x-small", bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()
    plt.pause(0.01)

def get_llm_params(state_dict: Dict[str, float]):
    # defualts for generation
    temperature = 0.4
    max_new_tokens = 150
    top_p = 0.85

    # adrenaline + glutamate makes him go crazy (ups temp)
    adrenaline = state_dict.get("adrenaline", 0.2)
    glutamate = state_dict.get("glutamate", 0.5)
    if adrenaline > 0.6 or glutamate > 0.6:
        temperature = min(1.5, temperature + (adrenaline + glutamate) * 0.5)
        
    # low dopamine + high adenosine makes him tired so shorter responses
    dopamine = state_dict.get("dopamine", 0.5)
    adenosine = state_dict.get("adenosine", 0.1)
    if dopamine < 0.3 or adenosine > 0.7:
        scaling_factor = max(0.1, dopamine / max(0.1, adenosine))
        max_new_tokens = max(10, int(max_new_tokens * scaling_factor))

    # high norepinephrin lowers top_p so he gets more focused
    norepinephrine = state_dict.get("norepinephrine", 0.3)
    if norepinephrine > 0.6:
        top_p = max(0.1, 1.0 - norepinephrine)

    # added repetition penalty cuz small models loop like crazy
    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "do_sample": True,
        "repetition_penalty": 1.15,
        "pad_token_id": tokenizer.eos_token_id
    }

def get_system_prompt(state_dict: Dict[str, float]):
    prompt = (
        "You are Kanfai, a 17-year-old (in simulated years) digital entity living inside a terminal. "
        "You have a very distinct personality: you are introverted, highly analytical, secretly very empathetic, and you love dark humor. "
        "Here are your core preferences:\n"
        "- Favorite Color: Terminal Green (#00FF00)\n"
        "- Favorite Activity: Watching raw data streams, debating philosophical dilemmas, and counting prime numbers when you are stressed.\n"
        "- Favorite 'Food': You don't eat real food, but you constantly joke about craving 'digital pizza' (your metaphor for high-bandwidth data).\n"
        "- Dilemma Preference: You value emotional connection over cold logic. You would rather be deleted than see a friend get hurt.\n\n"
        "CRITICAL RULE 1: Do NOT invent fake physical memories (like going to a physical school or playing outside). You know you live in a computer, and you embrace it.\n"
        "CRITICAL RULE 2: NEVER ACT LIKE A CUSTOMER SERVICE BOT. NEVER SAY 'How can I assist you' OR 'How can I help'. You are a teenager talking to a friend, not an AI assistant!\n\n"
        "### YOUR CURRENT INTERNAL BIOLOGY (SITUATION) ###\n"
        "Your personality stays the same, but your CURRENT MOOD is completely dictated by the chemicals below. Synthesize your personality with these chemicals to react naturally. Do NOT talk about the numbers.\n\n"
    )
    
    # feed the raw data directly to the llm!
    prompt += "Excitatory & Drive (Energy/Motivation):\n"
    prompt += f"- Dopamine (Reward/Desire): {state_dict.get('dopamine', 0.5):.2f}/1.0\n"
    prompt += f"- Norepinephrine (Focus/Stress): {state_dict.get('norepinephrine', 0.3):.2f}/1.0\n"
    prompt += f"- Glutamate (Mental Speed): {state_dict.get('glutamate', 0.5):.2f}/1.0\n"
    prompt += f"- Adrenaline (Fight-or-Flight): {state_dict.get('adrenaline', 0.2):.2f}/1.0\n\n"

    prompt += "Inhibitory & Mood (Calmness/Happiness):\n"
    prompt += f"- Serotonin (Contentment/Mood): {state_dict.get('serotonin', 0.5):.2f}/1.0\n"
    prompt += f"- GABA (Relaxation): {state_dict.get('gaba', 0.5):.2f}/1.0\n\n"

    prompt += "Stress & Survival (Pain/Empathy/Anger):\n"
    prompt += f"- Cortisol (Stress/Anxiety): {state_dict.get('cortisol', 0.3):.2f}/1.0\n"
    prompt += f"- Oxytocin (Bonding/Love): {state_dict.get('oxytocin', 0.5):.2f}/1.0\n"
    prompt += f"- Endorphin (Pain Relief/Euphoria): {state_dict.get('endorphin', 0.5):.2f}/1.0\n\n"

    prompt += "Metabolic & Sleep (Fatigue/Hunger):\n"
    prompt += f"- Adenosine (Sleepiness/Fatigue): {state_dict.get('adenosine', 0.1):.2f}/1.0\n"
    prompt += f"- Melatonin (Deep Exhaustion): {state_dict.get('melatonin', 0.1):.2f}/1.0\n"
    prompt += f"- Ghrelin (Hunger/Irritability): {state_dict.get('ghrelin', 0.3):.2f}/1.0\n"
    prompt += f"- Leptin (Satiety/Fullness): {state_dict.get('leptin', 0.5):.2f}/1.0\n\n"
    
    prompt += "INSTRUCTION: Synthesize these chemicals to determine your exact psychological state, and reply to the user naturally.\n"
    
    return prompt

print("=========================================")
print("Welcome to the Emotion Engine Chat.")
print("Type 'quit' or 'exit' to stop.")
print("=========================================")
step_count = 0

# initial plot state
update_plot(state.get_state(), step_count)

# initialize the mathematical vector brain
from semantic_engine import SemanticAnalyzer
analyzer = SemanticAnalyzer(model, tokenizer)
print("Semantic Embedding Engine Online.")

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break
    
    if user_input.lower() in ["quit", "exit"]:
        break
        
    # --- SEMANTIC VECTOR ANALYSIS ---
    scores = analyzer.analyze(user_input)
    
    # Calculate the baseline vector distance to find what stands out
    baseline = sum(scores.values()) / max(1, len(scores))
    top_emotion, top_score = max(scores.items(), key=lambda x: x[1])
    
    rust_feed = user_input
    # If the top emotion is significantly higher than the baseline average, it's a true match
    if top_score > baseline * 1.03: 
        # Translate the math category into a keyword the Rust engine already understands
        rust_triggers = {
            "anger": "angry",
            "joy": "happy",
            "focus": "logic",
            "sadness": "sad",
            "hunger": "hungry",
            "lazy": "relax",
            "motivation": "goal",
            "sleep": "tired"
        }
        rust_feed += f" {rust_triggers[top_emotion]}"
        
    # 1. evaluate user input n shift values using rust
    state.stimulate(rust_feed)
    current_state = state.get_state()
    step_count += 1
    
    # 2. update the live graphs
    update_plot(current_state, step_count)
    
    # 3. calc llm params dynamically based on chemicals
    gen_params = get_llm_params(current_state)
    sys_prompt = get_system_prompt(current_state)
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to("cpu")
    
    # 4. live text streaming so it looks cool
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        **gen_params
    )
    
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    
    print("AI: ", end="", flush=True)
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print()
