from collections import deque
import math
import os
import sys
import threading
from typing import Dict

import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
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

# Toggle between Qwen3.5 MoE (35B-A3B, 3B active per token) and Dense Qwen3.5 27B (both Q4 4-bit on A100 40GB)
USE_MOE_MODEL = True
QWEN35_MOE_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated"
QWEN35_27B_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-27B-abliterated"
LOAD_IN_4BIT = True


def load_kanfai_model():
    has_cuda = torch.cuda.is_available()
    if USE_REDDIT_BRAIN and os.path.exists("kanfai-360m"):
        chosen_id = "kanfai-360m"
        print(f"Loading REDDIT KANFAI ({chosen_id})...")
    elif USE_MOE_MODEL:
        chosen_id = QWEN35_MOE_MODEL_ID
        print(f"Loading KANFAI MoE ({chosen_id}) [Q4={LOAD_IN_4BIT and has_cuda}]...")
    else:
        chosen_id = QWEN35_27B_MODEL_ID
        print(f"Loading KANFAI 27B ({chosen_id}) [Q4={LOAD_IN_4BIT and has_cuda}]...")

    tok = AutoTokenizer.from_pretrained(chosen_id, trust_remote_code=True)

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if has_cuda:
        load_kwargs["device_map"] = "auto"
        if LOAD_IN_4BIT:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

    try:
        mdl = AutoModelForCausalLM.from_pretrained(chosen_id, **load_kwargs)
    except ValueError:
        # Qwen3.5 conditional generation architectures (qwen3_5 / qwen3_5_moe) register under ImageTextToText in some transformers versions
        mdl = AutoModelForImageTextToText.from_pretrained(chosen_id, **load_kwargs)

    if not has_cuda:
        mdl = mdl.to("cpu")
    mdl.eval()
    return chosen_id, tok, mdl


model_id, tokenizer, model = load_kanfai_model()

# --- TERMINAL UI SETUP ---
try:
    from terminal_ui import TerminalUI
    ui = TerminalUI()
except ImportError:
    print("Please pip install rich matplotlib")
    sys.exit(1)

# initialize the mathematical vector brain
from semantic_engine import SemanticAnalyzer
analyzer = SemanticAnalyzer(model, tokenizer)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def get_llm_params(state_dict: Dict[str, float]):
    # Extract excitatory, inhibitory, cognitive, and fatigue chemicals
    adrenaline = state_dict.get("adrenaline", 0.2)
    glutamate = state_dict.get("glutamate", 0.5)
    histamine = state_dict.get("histamine", 0.3)
    gaba = state_dict.get("gaba", 0.5)
    glycine = state_dict.get("glycine", 0.5)

    dopamine = state_dict.get("dopamine", 0.5)
    dhea = state_dict.get("dhea", 0.5)
    adenosine = state_dict.get("adenosine", 0.1)
    melatonin = state_dict.get("melatonin", 0.1)

    norepinephrine = state_dict.get("norepinephrine", 0.3)
    acetylcholine = state_dict.get("acetylcholine", 0.5)

    # 1. Smooth continuous temperature: driven upward by excitatory arousal (adrenaline/glutamate/histamine)
    #    and dampened by inhibitory calm (GABA/glycine). Baseline is ~0.55.
    net_arousal = (0.45 * adrenaline + 0.35 * glutamate + 0.20 * histamine) - (
        0.35 * gaba + 0.15 * glycine
    )
    temperature = 0.25 + 1.15 * _sigmoid(4.0 * (net_arousal - 0.05))
    temperature = max(0.15, min(1.40, round(temperature, 3)))

    # 2. Smooth continuous max_new_tokens: driven by motivation (dopamine/DHEA) vs fatigue (adenosine/melatonin)
    drive = 0.65 * dopamine + 0.35 * dhea
    fatigue = 0.60 * adenosine + 0.40 * melatonin
    energy_ratio = _sigmoid(4.5 * (drive - fatigue - 0.05))
    max_new_tokens = int(round(20 + 160 * energy_ratio))

    # 3. Smooth continuous top_p: high focus (acetylcholine + norepinephrine) narrows nucleus sampling
    focus_drive = 0.55 * norepinephrine + 0.45 * acetylcholine
    top_p = 0.95 - 0.65 * _sigmoid(5.0 * (focus_drive - 0.55))
    top_p = max(0.20, min(0.95, round(top_p, 3)))

    return {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "do_sample": True,
        "repetition_penalty": 1.15,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id
    }


STATIC_PERSONA_PROMPT = (
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
    "Your personality stays the same, but your CURRENT MOOD is completely dictated by the 20 chemicals below. Synthesize your personality with these chemicals to react naturally. Do NOT talk about the numbers.\n\n"
)


def get_system_prompt(state_dict: Dict[str, float]) -> str:
    return (
        f"{STATIC_PERSONA_PROMPT}"
        "Excitatory & Drive (Energy/Motivation/Alertness):\n"
        f"- Dopamine (Reward/Desire): {state_dict.get('dopamine', 0.5):.2f}/1.0\n"
        f"- Norepinephrine (Arousal/Vigilance): {state_dict.get('norepinephrine', 0.3):.2f}/1.0\n"
        f"- Glutamate (Mental Speed): {state_dict.get('glutamate', 0.5):.2f}/1.0\n"
        f"- Histamine (Wakefulness/Reactivity): {state_dict.get('histamine', 0.3):.2f}/1.0\n"
        f"- Adrenaline (Fight-or-Flight): {state_dict.get('adrenaline', 0.2):.2f}/1.0\n\n"
        "Inhibitory & Mood (Calmness/Stability/Resilience):\n"
        f"- Serotonin (Contentment/Mood): {state_dict.get('serotonin', 0.5):.2f}/1.0\n"
        f"- GABA (Relaxation/Calm): {state_dict.get('gaba', 0.5):.2f}/1.0\n"
        f"- Glycine (Motor/Mental Composure): {state_dict.get('glycine', 0.5):.2f}/1.0\n"
        f"- Prolactin (Social Satisfaction): {state_dict.get('prolactin', 0.3):.2f}/1.0\n"
        f"- DHEA (Stress Resilience/Drive): {state_dict.get('dhea', 0.5):.2f}/1.0\n\n"
        "Stress, Focus & Survival (Pain/Empathy/Cognition):\n"
        f"- Cortisol (Stress/Anxiety): {state_dict.get('cortisol', 0.3):.2f}/1.0\n"
        f"- Oxytocin (Bonding/Trust): {state_dict.get('oxytocin', 0.5):.2f}/1.0\n"
        f"- Vasopressin (Protective Loyalty): {state_dict.get('vasopressin', 0.3):.2f}/1.0\n"
        f"- Endorphin (Pain Relief/Euphoria): {state_dict.get('endorphin', 0.5):.2f}/1.0\n"
        f"- Acetylcholine (Deep Analytical Focus): {state_dict.get('acetylcholine', 0.5):.2f}/1.0\n\n"
        "Metabolic & Sleep (Fatigue/Hunger):\n"
        f"- Adenosine (Sleepiness/Fatigue): {state_dict.get('adenosine', 0.1):.2f}/1.0\n"
        f"- Melatonin (Deep Exhaustion): {state_dict.get('melatonin', 0.1):.2f}/1.0\n"
        f"- Ghrelin (Hunger/Irritability): {state_dict.get('ghrelin', 0.3):.2f}/1.0\n"
        f"- Leptin (Satiety/Fullness): {state_dict.get('leptin', 0.5):.2f}/1.0\n"
        f"- Insulin (Metabolic Satisfaction): {state_dict.get('insulin', 0.5):.2f}/1.0\n\n"
        "INSTRUCTION: Synthesize these chemicals to determine your exact psychological state, and reply to the user naturally.\n"
    )


def _run_generation(**kwargs):
    with torch.inference_mode():
        model.generate(**kwargs)


step_count = 0
ui.update_plot(state.get_state(), step_count)
semantic_enabled = True
chat_history = deque(maxlen=6)  # Keep last 6 turns of user/assistant exchanges for conversational continuity

while True:
    try:
        user_input = ui.get_input().strip()
    except EOFError:
        break

    # --- COMMAND HANDLER ---
    if user_input.startswith("/"):
        command = user_input.lower()
        if command in ["/exit", "/quit", "/q"]:
            print("Shutting down Kanfai OS... Flushing memory...")
            break
        elif command == "/semantic":
            semantic_enabled = not semantic_enabled
            status = "ONLINE" if semantic_enabled else "OFFLINE"
            print(f"[SYSTEM] Semantic Vector Engine is now {status}.")
            continue
        elif command == "/clear":
            chat_history.clear()
            print("[SYSTEM] Short-term conversational memory cleared.")
            continue
        else:
            print(f"[SYSTEM] Unknown command: {command}")
            continue

    if not user_input:
        continue

    # 1. Evaluate user input & shift chemical state using continuous semantic scores + lexical analysis
    if semantic_enabled and hasattr(state, "stimulate_with_semantics"):
        scores = analyzer.analyze(user_input)
        state.stimulate_with_semantics(user_input, scores)
    else:
        state.stimulate(user_input)

    current_state = state.get_state()
    step_count += 1

    # 2. Update the live graphs
    ui.update_plot(current_state, step_count)

    # 3. Calculate LLM params & system prompt dynamically based on all 20 chemicals
    gen_params = get_llm_params(current_state)
    sys_prompt = get_system_prompt(current_state)

    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})

    try:
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(**inputs, streamer=streamer, **gen_params)

    thread = threading.Thread(target=_run_generation, kwargs=gen_kwargs)
    thread.start()

    ui.print_ai_start()
    response_chunks = []
    for new_text in streamer:
        ui.print_ai_chunk(new_text)
        response_chunks.append(new_text)
    ui.print_ai_end()
    thread.join()

    assistant_reply = "".join(response_chunks).strip()
    if assistant_reply:
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": assistant_reply})
