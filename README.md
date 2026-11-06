# Kanfai: Biologically Accurate AI 

yo! welcome to the **kanfai** project by NGSTRIKER. i built this cuz i got bored of normal ai being so fake and wanted to see what happens if we give an ai an actual simulated brain chemistry. 

instead of just prompting an ai to "act happy," kanfai physically tracks 20 different neurotransmitters in real time. his emotional state is mathematically slaved to these chemicals. if u stress him out, his cortisol spikes and he gets defensive. if u talk for too long, his adenosine builds up and he literally falls asleep.

---

## The Architecture (How it actually works)

this project is built on a 3-part neuro-symbolic stack. it runs locally or accelerated on GPUs (e.g., A100 40GB).

```
   [User Input] 
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 1. Semantic Vector Brain (Python / PyTorch)             │
 │    • Anisotropy mean-centering (All-but-the-Top)        │
 │    • Norm-weighted pooling & negation look-back         │
 │    • Single-pass vectorized cosine similarity (8 anchors)│
 └────────────────────────────┬────────────────────────────┘
                              │ Continuous emotion z-scores
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 2. Biological Engine (Rust + PyO3)                      │
 │    • 20 tracked neurotransmitters & hormones            │
 │    • Continuous sensitivity matrix W (20 x 8)           │
 │    • Coupled homeostasis & cross-chemical antagonism     │
 │    • Asymptotic soft saturation (no linear clipping)    │
 │    • Punctuation- & negation-aware lexical matching     │
 └────────────────────────────┬────────────────────────────┘
                              │ 20 chemical values
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ 3. Uncensored LLM (Qwen 3.5 MoE / 27B / 1.5B)           │
 │    • Ingests all 20 chemicals into system prompt        │
 │    • Continuous sigmoid sampling (temp, top_p, length)   │
 │    • Abliterated (zero customer-service refusals)       │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
                    [Real-time Streamed Voice]
```

### 1. The Semantic Vector Brain (Python)
when u type a message, we don't just search for keywords. we use a massive brain-hack: we take the user's sentence and run it through the model's raw **embedding layer** (`model.get_input_embeddings()`). this bypasses the heavy generation pass and extracts the semantic coordinates in milliseconds.
- **Anisotropy correction:** causal LLM embeddings cluster in a narrow cone; we subtract the global vocabulary mean vector to center the space.
- **Norm-weighted pooling & negation:** informative tokens are weighted by their centered L2 norm, stop words are down-weighted, and words preceded by negations (*"not happy"*) are inverted.
- **Vectorized anchor projection:** sentence vectors are multiplied against an $(8 \times D)$ anchor matrix to compute continuous cosine similarities across 8 emotional dimensions (*Anger, Joy, Focus, Sadness, Hunger, Lazy, Motivation, Sleep*).

### 2. The Biological Engine (Rust)
the semantic engine pipes continuous z-scores directly into a blazing-fast **Rust** matrix compiled via PyO3:
- **20 Neurotransmitters & Hormones:** Dopamine, Serotonin, Norepinephrine, Acetylcholine, GABA, Glutamate, Glycine, Histamine, Adenosine, Cortisol, Adrenaline, Oxytocin, Melatonin, Endorphin, Vasopressin, Ghrelin, Leptin, Insulin, Prolactin, DHEA.
- **Continuous Sensitivity Matrix ($W \in \mathbb{R}^{20 \times 8}$):** transforms continuous emotional activations into specific chemical impulses.
- **Coupled Homeostasis & Cross-Antagonism:**
  - Fast catecholamines (adrenaline, glutamate) clear quickly, while steroids/peptides (cortisol, oxytocin) linger.
  - Cortisol suppresses serotonin, oxytocin, and DHEA; GABA inhibits glutamate and adrenaline; oxytocin buffers stress; ghrelin & leptin trigger mutual hunger/satiety dynamics.
- **Asymptotic Soft Saturation:** values smoothly approach boundaries without abrupt linear clipping at 0.0 or 1.0.

### 3. The Uncensored Model (Qwen 3.5 MoE & 27B)
it is CRITICAL that we use an "abliterated" / uncensored model. normal models have heavy RLHF that forces them to act like polite corporate assistants.
- **Default GPU Model:** `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` (MoE with 35B total parameters, 3B active per token — fast 120+ tok/s on an A100 GPU in 4-bit NF4).
- **Dense 27B Model:** `huihui-ai/Huihui-Qwen3.5-27B-abliterated` (switchable via `USE_MOE_MODEL = False` in `app.py`).
- **Dynamic Sampling Modulations:** continuous sigmoids modulate generation on every turn:
  - High arousal (adrenaline + glutamate) smoothly boosts `temperature`.
  - Mental focus (acetylcholine + norepinephrine) narrows `top_p`.
  - Fatigue (adenosine + melatonin) smoothly reduces response length (`max_new_tokens`).

For an in-depth mathematical breakdown, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## ⚡ Google Colab (A100 GPU)

You can run Kanfai on an **A100 GPU** in Google Colab using the included notebook:
1. Open **[kanfai_colab.ipynb](kanfai_colab.ipynb)** in Google Colab.
2. Select **Runtime** > **Change runtime type** > **A100 GPU** (High-RAM).
3. Run the setup cells to install Rust, dependencies, and compile the engine.
4. Launch `%run app.py` and chat interactively!

---

## Local Getting Started

Make sure you have Python (3.8+) and the Rust compiler installed (`cargo --version`).

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install --upgrade maturin
```

### 2. Compile the Rust Engine (Mandatory)
Compile the biological engine into a native library for Python:

**Linux / macOS:**
```bash
cd emotion_engine
PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1" maturin develop --release
cd ..
```

**Windows (PowerShell):**
```powershell
cd emotion_engine
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1"
python -m maturin develop --release
cd ..
```

### 3. Boot up Kanfai
```bash
python app.py
```

### Terminal Commands inside Kanfai:
- `/semantic` — toggle the semantic vector engine ON/OFF
- `/clear` — flush short-term conversational chat history
- `/q` or `/exit` — save state and exit

---

have fun playing with him. be nice or he will get depressed and hangry.
