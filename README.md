# Kanfai: Biologically Accurate AI 

yo! welcome to the **kanfai** project by NGSTRIKER. i built this cuz i got bored of normal ai being so fake and wanted to see what happens if we give an ai an actual simulated brain chemistry. 

instead of just prompting an ai to "act happy," kanfai physically tracks 20 different neurotransmitters in real time. his emotional state is mathematically slaved to these chemicals. if u stress him out, his cortisol spikes and he gets defensive. if u talk for too long, his adenosine builds up and he literally falls asleep.

## The Architecture (How it actually works)

this project is built on a 3-part neuro-symbolic stack. it's incredibly fast and runs entirely locally.

### 1. The Semantic Vector Brain (Python)
when u type a message, we don't just search for keywords. we use a massive brain-hack: we take the user's sentence and run it through the Qwen model's raw **embedding layer** (`model.model.embed_tokens`). this bypasses the heavy AI generation and just outputs the exact 3D mathematical coordinates of your sentence in milliseconds.
we then use **Cosine Similarity** to measure how close your sentence is to 8 emotional anchors (like Grief, Joy, Anger). this means the engine understands context without running a heavy secondary neural network.

### 2. The Biological Engine (Rust)
the semantic engine pipes its findings directly into a blazing-fast **Rust** matrix (compiled via PyO3). this engine tracks the 20 chemicals from 0.0 to 1.0. 
- **praise/bonding:** spikes oxytocin and dopamine.
- **threats/loss:** spikes cortisol and drops serotonin.
- **metabolic decay:** chemicals slowly clear out every turn, while sleep chemicals (adenosine/melatonin) naturally build up.

### 3. The Uncensored model (Qwen 2.5 1.5B)
we use `thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored`. it is CRITICAL that we use an "abliterated" / uncensored model. normal models have heavy RLHF (safety training) that forces them to act like polite customer service bots. 
by using an uncensored model, we completely strip out the "How can I assist you" guardrails. we feed the raw biological matrix (e.g., `Dopamine: 0.8, Cortisol: 0.2`) straight into the system prompt. the LLM reads its own chemical state, synthesizes it with its teenage personality sheet, and acts out the emotions flawlessly.

---

## Getting Started

make sure u got python (3.8+) and the rust compiler installed. 

### 1. Install stuff
```powershell
pip install -r requirements.txt
pip install --upgrade maturin
```

### 2. Compile the Rust Engine (Mandatory)
we need to compile the biological engine into a native C-library for python to use.
```powershell
cd emotion_engine
# forces compilation even if u are using newer python versions (like 3.14)
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1"
python -m maturin develop --release
cd ..
```

### 3. Boot up Kanfai
```powershell
python app.py
```
*(note: the first time u run this, it will download the 1.5B Qwen model from huggingface, which takes about 3-4 GB of space. make sure u have room!)*

have fun playing with him. be nice or he will get depressed and hangry. 
