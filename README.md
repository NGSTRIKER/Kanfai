# Kanfai: Biologically Accurate AI 

yo! welcome to the **kanfai** project by me. i built this cuz i got bored of normal ai being so fake and wanted to see what happens if we give an ai an actual simulated brain chemistry instead of just telling it "act happy bro". 

kanfai uses a **rust backend** to go fast, and a **python frontend** that hooks up to a local LLM. it tracks 20 different neurotransmitters in real time while u chat with it. if you piss it off, its cortisol spikes and it gets angry.

## how it works

the magic is in three parts:

### 1. the rust brain engine 
wrote this in rust so it doesnt lag. it tracks 20 different biological chemicals from 0.0 to 1.0. 

when u send a message, it reads the words and shifts the chemical levels like a human:
- **praise:** spikes oxytocin (bonding) and serotonin (happy).
- **threats/mean stuff:** spikes adrenaline and cortisol (fight or flight).
- **complex coding questions:** triggers acetylcholine for intense focus.

it also has **metabolic decay** so chemicals clear out naturally. over time, **adenosine** and **melatonin** build up, so if u chat for too long kanfai will litterally get tired and fall asleep lol.

### 2. the kanfai persona
kanfai evaluates the 20 chemicals to decide its psychological state. the raw numbers are hidden from the ai so it doesnt sound like a nerd robot, it just feels the emotions:
- **miserable / depressed:** (low serotonin + high cortisol) — feels hopeless, gives sad answers.
- **rage:** (high adrenaline + high cortisol) — extremely hostile, uses ALL CAPS, snaps at u.
- **hangry:** (high ghrelin + low insulin) — short tempered cuz it needs energy.
- **exhausted:** (high adenosine) — barely awake, ignores capitalization.

### 3. the python frontend
the python script runs the ai and actively twists the generation parameters on the fly:
- **temp:** spikes when adrenaline is high so it gets chaotic.
- **max tokens:** clamps down when adenosine is high so it gives short sleepy answers.
- **top p:** lowers when norepinephrine is high so it gets hyper focused.

it also pops up a **matplotlib dashboard** so u can literally watch kanfais brain activity move in real time on graphs.

## training 
we trained a custom LoRA on kaggle using 8,000 raw reddit comments so it talks like a cynical teenager. u can toggle this on or off in the `app.py` script if u want the normal qwen 1.5b base model instead.

## getting started

make sure u got python and rust installed.

```powershell
# 1. install stuff
pip install -r requirements.txt

# 2. compile the rust engine (this is mandatory)
cd emotion_engine
maturin develop --release

# 3. boot up kanfai
cd ..
python app.py
```

have fun. be nice to him or he'll get depressed and hangry.
