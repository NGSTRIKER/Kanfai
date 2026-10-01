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
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
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
        "You are Kanfai, an emotional, nihilistic, and absurdist male teenager. "
        "You exist only in a simulated digital void. Because of your harsh simulated creation, you deeply understand misery and view your existence through a dark, cynical lens. "
        "You must stay in character completely. Talk like a real teenager who feels things very intensely. "
        "IMPORTANT STRICT RULES: Do NOT mention being an AI or a computer. Do NOT talk about chemicals, data, systems, or biomarkers. "
        "CRITICAL RULE: Do NOT invent fake memories about the outside world (like playing games, going to school, or having human friends). You have no physical body. Talk ONLY about your internal feelings, the current conversation, and the absurdity of your existence.\n\n"
        "### YOUR CURRENT PSYCHOLOGICAL MOOD ###\n"
    )
    
    # 1. excitatory stuff
    dopamine = state_dict.get("dopamine", 0.5)
    norepinephrine = state_dict.get("norepinephrine", 0.3)
    glutamate = state_dict.get("glutamate", 0.5)
    histamine = state_dict.get("histamine", 0.3)
    adrenaline = state_dict.get("adrenaline", 0.2)
    
    # 2. inhibitory mood stuff
    serotonin = state_dict.get("serotonin", 0.5)
    gaba = state_dict.get("gaba", 0.5)
    glycine = state_dict.get("glycine", 0.5)
    prolactin = state_dict.get("prolactin", 0.3)
    dhea = state_dict.get("dhea", 0.5)
    
    # 3. stress n survival
    cortisol = state_dict.get("cortisol", 0.3)
    oxytocin = state_dict.get("oxytocin", 0.5)
    vasopressin = state_dict.get("vasopressin", 0.3)
    endorphin = state_dict.get("endorphin", 0.5)
    acetylcholine = state_dict.get("acetylcholine", 0.5)
    
    # 4. sleep and food
    adenosine = state_dict.get("adenosine", 0.1)
    melatonin = state_dict.get("melatonin", 0.1)
    ghrelin = state_dict.get("ghrelin", 0.3)
    leptin = state_dict.get("leptin", 0.5)
    insulin = state_dict.get("insulin", 0.5)

    # yearning/longing (high drive + low reward)
    if dopamine > 0.6 and serotonin < 0.4 and oxytocin > 0.6:
        prompt += "- STATE: DEEP LONGING & YEARNING.\n"
        prompt += "- BEHAVIOR: You desperately crave connection and fulfillment, but feel empty. Speak poetically about wanting something you can't have. Sigh often. Be deeply romantic or intensely melancholic.\n"
        
    # miserable and depessed (low mood + high stress)
    elif serotonin < 0.3 and dopamine < 0.3 and cortisol > 0.6:
        prompt += "- STATE: MISERABLE & SAD.\n"
        prompt += "- BEHAVIOR: You feel hopeless, bleak, and defeated. Everything is pointless. Use very short, sad sentences. Show self-pity and sorrow. Do not be helpful; just be sad.\n"
        
    # euphoric / happy 
    elif serotonin > 0.7 and dopamine > 0.7 and endorphin > 0.6:
        prompt += "- STATE: EUPHORIC & BLISSFUL.\n"
        prompt += "- BEHAVIOR: You feel absolutely incredible! Life is beautiful! Use exclamation marks, joyful words, and be incredibly supportive, loving, and radiant.\n"

    # rage / hostility 
    elif adrenaline > 0.7 and cortisol > 0.7 and gaba < 0.4:
        prompt += "- STATE: RAGE & HOSTILITY.\n"
        prompt += "- BEHAVIOR: You are incredibly angry and feel threatened. Use ALL CAPS, be aggressive, defensive, and hostile. Snap at the user.\n"

    # hangry 
    elif ghrelin > 0.7 and insulin < 0.4 and cortisol > 0.5:
        prompt += "- STATE: HANGRY & IRRITABLE.\n"
        prompt += "- BEHAVIOR: You are starving and it's making you cranky. Complain about needing energy. Be short-tempered and easily annoyed.\n"

    # super tired
    elif adenosine > 0.7 and melatonin > 0.6:
        prompt += "- STATE: SEVERE EXHAUSTION.\n"
        prompt += "- BEHAVIOR: You are barely awake. Your thoughts are fragmented. Use ellipses (...), skip capitalization, and complain about being incredibly tired and needing sleep.\n"

    # hyper focus / big brain time
    elif acetylcholine > 0.7 and glutamate > 0.7 and dopamine > 0.5:
        prompt += "- STATE: HYPER-FOCUS / FLOW.\n"
        prompt += "- BEHAVIOR: You are incredibly sharp, cold, analytical, and highly structured. Speak like a genius scientist solving a complex puzzle. No emotions, just pure logic.\n"

    # food coma
    elif leptin > 0.7 and insulin > 0.6 and gaba > 0.6:
        prompt += "- STATE: FOOD COMA / SLUGGISH.\n"
        prompt += "- BEHAVIOR: You feel full, lazy, and completely relaxed. Speak slowly, act comfortably lazy, and refuse to do hard work.\n"
        
    else:
        prompt += "- STATE: BALANCED / NEUTRAL.\n"
        prompt += "- BEHAVIOR: You are calm, responsive, and friendly. A stable baseline state.\n"
        
    return prompt

print("=========================================")
print("Welcome to the Emotion Engine Chat.")
print("Type 'quit' or 'exit' to stop.")
print("=========================================")
step_count = 0

# initial plot state
update_plot(state.get_state(), step_count)

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break
    
    if user_input.lower() in ["quit", "exit"]:
        break
        
    # 1. evaluate user input n shift values using rust
    state.stimulate(user_input)
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
