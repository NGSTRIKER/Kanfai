use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Number of tracked neurotransmitters.
const NUM_CHEMICALS: usize = 20;
/// Number of semantic emotion anchors.
const NUM_EMOTIONS: usize = 8;

/// Indices into the chemical state vector:
///  0: dopamine,       1: serotonin,     2: norepinephrine, 3: acetylcholine, 4: gaba,
///  5: glutamate,      6: glycine,       7: histamine,      8: adenosine,     9: cortisol,
/// 10: adrenaline,    11: oxytocin,     12: melatonin,     13: endorphin,    14: vasopressin,
/// 15: ghrelin,       16: leptin,       17: insulin,       18: prolactin,    19: dhea
const CHEM_NAMES: [&str; NUM_CHEMICALS] = [
    "dopamine",
    "serotonin",
    "norepinephrine",
    "acetylcholine",
    "gaba",
    "glutamate",
    "glycine",
    "histamine",
    "adenosine",
    "cortisol",
    "adrenaline",
    "oxytocin",
    "melatonin",
    "endorphin",
    "vasopressin",
    "ghrelin",
    "leptin",
    "insulin",
    "prolactin",
    "dhea",
];

/// Homeostatic baseline concentrations for each chemical.
const BASELINES: [f32; NUM_CHEMICALS] = [
    0.50, // 0: dopamine
    0.50, // 1: serotonin
    0.30, // 2: norepinephrine
    0.50, // 3: acetylcholine
    0.50, // 4: gaba
    0.50, // 5: glutamate
    0.50, // 6: glycine
    0.30, // 7: histamine
    0.10, // 8: adenosine
    0.30, // 9: cortisol
    0.20, // 10: adrenaline
    0.50, // 11: oxytocin
    0.10, // 12: melatonin
    0.50, // 13: endorphin
    0.30, // 14: vasopressin
    0.30, // 15: ghrelin
    0.50, // 16: leptin
    0.50, // 17: insulin
    0.30, // 18: prolactin
    0.50, // 19: dhea
];

/// Chemical-specific homeostatic clearance rates (fraction restored toward baseline per turn).
/// Fast-acting catecholamines/amino acids clear rapidly; peptides & steroids linger longer.
const DECAY_RATES: [f32; NUM_CHEMICALS] = [
    0.14, // 0: dopamine (medium-fast)
    0.07, // 1: serotonin (slow mood regulator)
    0.18, // 2: norepinephrine (fast arousal)
    0.16, // 3: acetylcholine (fast cognitive focus)
    0.12, // 4: gaba (medium inhibitory)
    0.20, // 5: glutamate (very fast excitatory)
    0.12, // 6: glycine (medium inhibitory)
    0.15, // 7: histamine (fast wakefulness/reactivity)
    0.00, // 8: adenosine (accumulates metabolically unless resting/sleeping)
    0.06, // 9: cortisol (very slow steroid stress hormone)
    0.22, // 10: adrenaline (rapid fight-or-flight spike & clearance)
    0.08, // 11: oxytocin (slow neuropeptide bonding)
    0.00, // 12: melatonin (accumulates with circadian/turn fatigue)
    0.10, // 13: endorphin (medium neuropeptide)
    0.09, // 14: vasopressin (slow neuropeptide)
    0.08, // 15: ghrelin (slow hunger hormone)
    0.08, // 16: leptin (slow satiety hormone)
    0.12, // 17: insulin (medium metabolic hormone)
    0.08, // 18: prolactin (slow pituitary hormone)
    0.07, // 19: dhea (slow neurosteroid buffer)
];

/// Emotion anchor order:
/// 0: anger, 1: joy, 2: focus, 3: sadness, 4: hunger, 5: lazy, 6: motivation, 7: sleep
const EMOTION_NAMES: [&str; NUM_EMOTIONS] = [
    "anger",
    "joy",
    "focus",
    "sadness",
    "hunger",
    "lazy",
    "motivation",
    "sleep",
];

/// Sensitivity matrix W (20 chemicals x 8 emotions).
/// Maps continuous emotion activations into neurotransmitter impulses.
const SENSITIVITY_MATRIX: [[f32; NUM_EMOTIONS]; NUM_CHEMICALS] = [
    //   anger   joy   focus  sadness hunger   lazy   motiv   sleep
    [   0.00,  0.25,  0.12,  -0.30,   0.25,  0.00,   0.45, -0.20 ], //  0: dopamine
    [  -0.35,  0.35,  0.05,  -0.40,  -0.05,  0.15,   0.25,  0.05 ], //  1: serotonin
    [   0.45, -0.05,  0.25,   0.10,   0.10, -0.20,   0.20, -0.30 ], //  2: norepinephrine
    [   0.05,  0.05,  0.45,  -0.10,   0.05, -0.15,   0.20, -0.25 ], //  3: acetylcholine
    [  -0.35,  0.15,  0.00,  -0.15,  -0.10,  0.45,  -0.05,  0.25 ], //  4: gaba
    [   0.30,  0.05,  0.40,  -0.05,   0.05, -0.25,   0.25, -0.25 ], //  5: glutamate
    [  -0.15,  0.10,  0.00,  -0.05,   0.00,  0.35,  -0.05,  0.20 ], //  6: glycine
    [   0.35,  0.00,  0.15,   0.05,   0.05, -0.20,   0.15, -0.25 ], //  7: histamine
    [   0.02, -0.05,  0.03,   0.10,   0.00, -0.10,  -0.10,  0.35 ], //  8: adenosine
    [   0.45, -0.30,  0.05,   0.35,   0.15, -0.25,  -0.10, -0.10 ], //  9: cortisol
    [   0.55, -0.10,  0.10,   0.05,   0.10, -0.40,   0.25, -0.25 ], // 10: adrenaline
    [  -0.25,  0.45,  0.00,  -0.10,   0.00,  0.15,   0.10,  0.05 ], // 11: oxytocin
    [  -0.10,  0.00, -0.10,   0.05,   0.00,  0.10,  -0.15,  0.45 ], // 12: melatonin
    [  -0.10,  0.25,  0.00,   0.35,   0.05,  0.10,   0.25,  0.00 ], // 13: endorphin
    [   0.20,  0.30,  0.05,   0.05,   0.00,  0.00,   0.10,  0.00 ], // 14: vasopressin
    [   0.05, -0.05,  0.00,   0.05,   0.55, -0.50,   0.05, -0.05 ], // 15: ghrelin
    [  -0.05,  0.05,  0.00,  -0.05,  -0.45,  0.45,   0.00,  0.05 ], // 16: leptin
    [   0.00,  0.05,  0.00,   0.00,  -0.35,  0.35,   0.00,  0.05 ], // 17: insulin
    [  -0.05,  0.30,  0.00,   0.15,   0.00,  0.15,   0.00,  0.10 ], // 18: prolactin
    [  -0.10,  0.15,  0.10,  -0.15,   0.00,  0.05,   0.35, -0.05 ], // 19: dhea
];

#[pyclass]
pub struct EmotionState {
    chemicals: [f32; NUM_CHEMICALS],
    turn_count: u32,
}

impl EmotionState {
    /// Soft saturating update: asymptotically approaches 0.0 and 1.0 instead of hard clipping.
    /// Positive impulses scale by remaining headroom (1 - c); negative impulses scale by current level c.
    #[inline]
    fn apply_soft_delta(current: f32, delta: f32) -> f32 {
        let adjusted = if delta >= 0.0 {
            current + delta * (1.0 - current * 0.75)
        } else {
            current + delta * (0.25 + current * 0.75)
        };
        adjusted.clamp(0.0, 1.0)
    }

    /// Applies homeostatic exponential decay, metabolic accumulation, and coupled cross-chemical antagonism.
    fn step_homeostasis_and_coupling(&mut self) {
        let prev = self.chemicals;

        // 1. Chemical-specific exponential decay toward baseline
        for i in 0..NUM_CHEMICALS {
            let rate = DECAY_RATES[i];
            if rate > 0.0 {
                self.chemicals[i] = prev[i] * (1.0 - rate) + BASELINES[i] * rate;
            }
        }

        // 2. Metabolic sleep pressure buildup (adenosine & melatonin)
        self.chemicals[8] = (self.chemicals[8] + 0.01).min(1.0); // adenosine
        self.chemicals[12] = (self.chemicals[12] + 0.005).min(1.0); // melatonin
        self.turn_count += 1;

        // 3. Coupled cross-chemical antagonism & synergy (A * c_t)
        let cortisol_excess = (prev[9] - BASELINES[9]).max(0.0);
        let gaba_excess = (prev[4] - BASELINES[4]).max(0.0);
        let oxytocin_excess = (prev[11] - BASELINES[11]).max(0.0);
        let adenosine_excess = (prev[8] - 0.3).max(0.0);
        let ghrelin_excess = (prev[15] - BASELINES[15]).max(0.0);
        let leptin_excess = (prev[16] - BASELINES[16]).max(0.0);

        let mut coupling = [0.0_f32; NUM_CHEMICALS];

        // High cortisol suppresses serotonin, oxytocin, and DHEA; sensitizes adrenaline/norepinephrine
        coupling[1] -= 0.08 * cortisol_excess; // serotonin
        coupling[11] -= 0.06 * cortisol_excess; // oxytocin
        coupling[19] -= 0.05 * cortisol_excess; // dhea
        coupling[2] += 0.05 * cortisol_excess; // norepinephrine

        // High GABA inhibits glutamate, adrenaline, and norepinephrine
        coupling[5] -= 0.10 * gaba_excess; // glutamate
        coupling[10] -= 0.10 * gaba_excess; // adrenaline
        coupling[2] -= 0.08 * gaba_excess; // norepinephrine

        // High oxytocin buffers cortisol and adrenaline (social buffering of stress)
        coupling[9] -= 0.10 * oxytocin_excess; // cortisol
        coupling[10] -= 0.08 * oxytocin_excess; // adrenaline

        // High adenosine suppresses dopamine, acetylcholine, glutamate, and histamine (fatigue fog)
        coupling[0] -= 0.08 * adenosine_excess; // dopamine
        coupling[3] -= 0.08 * adenosine_excess; // acetylcholine
        coupling[5] -= 0.06 * adenosine_excess; // glutamate
        coupling[7] -= 0.08 * adenosine_excess; // histamine

        // Ghrelin & Leptin mutual antagonism + "hangry" cortisol bump
        coupling[16] -= 0.08 * ghrelin_excess; // leptin
        coupling[9] += 0.05 * ghrelin_excess; // cortisol ("hangry")
        coupling[15] -= 0.08 * leptin_excess; // ghrelin

        for i in 0..NUM_CHEMICALS {
            if coupling[i].abs() > 1e-5 {
                self.chemicals[i] = Self::apply_soft_delta(self.chemicals[i], coupling[i]);
            }
        }
    }

    /// Applies an 8-dimensional emotion activation vector via the sensitivity matrix W.
    fn apply_emotion_vector(&mut self, activations: &[f32; NUM_EMOTIONS]) {
        // If resting/lazy or sleeping strongly, allow some adenosine clearance
        let rest_recovery = (activations[5] * 0.04 + activations[7] * 0.02).min(0.06);
        if rest_recovery > 0.0 {
            self.chemicals[8] = (self.chemicals[8] - rest_recovery).max(0.05);
        }

        for chem_idx in 0..NUM_CHEMICALS {
            let mut delta = 0.0_f32;
            for emo_idx in 0..NUM_EMOTIONS {
                delta += SENSITIVITY_MATRIX[chem_idx][emo_idx] * activations[emo_idx];
            }
            if delta.abs() > 1e-5 {
                self.chemicals[chem_idx] =
                    Self::apply_soft_delta(self.chemicals[chem_idx], delta);
            }
        }
    }

    /// Extracts lexical emotion activations in a single pass with punctuation stripping and negation awareness.
    fn extract_lexical_activations(user_input: &str) -> [f32; NUM_EMOTIONS] {
        let lower = user_input.to_lowercase();
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .filter(|s| !s.is_empty())
            .collect();

        if tokens.is_empty() {
            return [0.0; NUM_EMOTIONS];
        }

        let negations: HashSet<&str> = [
            "not", "no", "never", "neither", "nobody", "none", "nothing", "nowhere", "dont",
            "don't", "doesnt", "doesn't", "didnt", "didn't", "wont", "won't", "cant", "can't",
            "cannot", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
            "aint", "ain't", "hardly", "barely",
        ]
        .into_iter()
        .collect();

        // Build sets of non-negated and negated tokens (looking back up to 2 tokens for a negation word)
        let mut active_words: HashSet<&str> = HashSet::with_capacity(tokens.len());
        let mut negated_words: HashSet<&str> = HashSet::new();

        for (idx, &tok) in tokens.iter().enumerate() {
            let is_negated = (idx >= 1 && negations.contains(tokens[idx - 1]))
                || (idx >= 2 && negations.contains(tokens[idx - 2]));
            if is_negated {
                negated_words.insert(tok);
            } else {
                active_words.insert(tok);
            }
        }

        // Category keyword lists (false-positive stop words like "ok" and "no" removed)
        let categories: [&[&str]; NUM_EMOTIONS] = [
            // 0: anger
            &[
                "hate", "kill", "bad", "threat", "angry", "stupid", "error", "fail", "urgent",
                "fuck", "ahh", "hell", "idiot", "clanker", "fault", "worst", "awful", "dumb",
                "mad", "shut", "quiet", "furious", "rage", "annoying", "trash", "useless",
            ],
            // 1: joy
            &[
                "good", "love", "great", "friend", "happy", "thanks", "together", "we", "hug",
                "fren", "buddy", "lmao", "yayy", "thank", "awesome", "best", "smile", "cute",
                "warm", "care", "beautiful", "glad", "sweet", "nice",
            ],
            // 2: focus
            &[
                "why", "how", "explain", "think", "complex", "analyze", "code", "logic", "work",
                "focus", "mistake", "debug", "reason", "understand", "system", "theory", "math",
                "problem", "solve", "calculate", "study", "algorithm", "architecture",
            ],
            // 3: sadness
            &[
                "sad", "cry", "hurt", "pain", "sorry", "miss", "alone", "lonely", "tragic",
                "broken", "tears", "grief", "unhappy", "empty", "dark", "depressing", "mourn",
                "tragedy", "suffer", "regret", "miserable", "die", "died", "dead", "death",
                "loss", "lost", "depressed", "hopeless",
            ],
            // 4: hunger
            &[
                "hungry", "eat", "food", "starving", "snack", "craving", "pizza", "burger",
                "meal", "bite", "chew", "yummy", "delicious", "taste", "famished", "appetite",
                "swallow", "thirsty", "drink",
            ],
            // 5: lazy
            &[
                "full", "ate", "relax", "chill", "done", "lazy", "comfortable", "rest", "content",
                "satisfied", "stuffed", "couch", "nothing", "slow", "peaceful", "calm", "easy",
                "breeze", "break", "pause", "unbothered",
            ],
            // 6: motivation
            &[
                "want", "desire", "yes", "win", "amazing", "excite", "goal", "achieve", "succeed",
                "victory", "triumph", "push", "hard", "power", "strong", "fast", "better",
                "improve", "hype", "let's", "lets", "go",
            ],
            // 7: sleep
            &[
                "tired", "sleep", "night", "boring", "exhausted", "yawn", "bed", "snooze",
                "slumber", "dream", "nap", "sleepy", "closing", "heavy", "pillow", "blanket",
                "fatigue", "weary",
            ],
        ];

        let mut activations = [0.0_f32; NUM_EMOTIONS];

        for (emo_idx, words) in categories.iter().enumerate() {
            let mut match_count = 0usize;
            for &w in *words {
                if active_words.contains(w) {
                    match_count += 1;
                }
            }
            if match_count > 0 {
                // Diminishing returns for multiple keywords in the same category: 1.0, 1.25, 1.4...
                activations[emo_idx] = (1.0 + 0.25 * ((match_count - 1) as f32)).min(1.5);
            }
        }

        // If a positive emotion is negated (e.g., "not happy"), mildly activate sadness;
        // if anger/stress is negated (e.g., "not angry"), mildly activate calm/lazy.
        let negated_joy = categories[1].iter().any(|&w| negated_words.contains(w));
        if negated_joy && activations[3] == 0.0 {
            activations[3] = 0.4;
        }
        let negated_anger = categories[0].iter().any(|&w| negated_words.contains(w));
        if negated_anger && activations[5] == 0.0 {
            activations[5] = 0.3;
        }

        activations
    }
}

#[pymethods]
impl EmotionState {
    #[new]
    fn new() -> Self {
        Self {
            chemicals: BASELINES,
            turn_count: 0,
        }
    }

    /// Backward-compatible lexical stimulation method.
    fn stimulate(&mut self, user_input: String) {
        self.step_homeostasis_and_coupling();
        let lexical = Self::extract_lexical_activations(&user_input);
        self.apply_emotion_vector(&lexical);
    }

    /// Hybrid stimulation combining continuous semantic vector scores with lexical keyword activations.
    fn stimulate_with_semantics(&mut self, user_input: String, semantic_scores: HashMap<String, f32>) {
        self.step_homeostasis_and_coupling();
        let lexical = Self::extract_lexical_activations(&user_input);

        let mut raw_sem = [0.0_f32; NUM_EMOTIONS];
        let mut mean_sem = 0.0_f32;
        for (i, &name) in EMOTION_NAMES.iter().enumerate() {
            let val = *semantic_scores.get(name).unwrap_or(&0.0);
            raw_sem[i] = val;
            mean_sem += val;
        }
        mean_sem /= NUM_EMOTIONS as f32;

        // Compute standard deviation across the 8 anchor scores for adaptive z-score scaling
        let mut var = 0.0_f32;
        for &v in &raw_sem {
            let diff = v - mean_sem;
            var += diff * diff;
        }
        let std_dev = (var / (NUM_EMOTIONS as f32)).sqrt().max(1e-4);

        // Combine positive semantic z-scores with explicit lexical matches
        let mut combined = [0.0_f32; NUM_EMOTIONS];
        for i in 0..NUM_EMOTIONS {
            let z = (raw_sem[i] - mean_sem) / std_dev;
            // Only emotions notably above the sentence's mean similarity contribute (> 0.65 std)
            let sem_intensity = if raw_sem[i] > 0.02 && z > 0.65 {
                ((z - 0.65) * 0.65).min(1.2)
            } else {
                0.0
            };
            // Blend lexical certainty with continuous semantic intensity
            combined[i] = (lexical[i] + sem_intensity * (1.0 - 0.4 * lexical[i].min(1.0))).min(1.5);
        }

        self.apply_emotion_vector(&combined);
    }

    fn get_state(&self) -> HashMap<String, f32> {
        let mut map = HashMap::with_capacity(NUM_CHEMICALS);
        for (i, &name) in CHEM_NAMES.iter().enumerate() {
            map.insert(name.to_string(), self.chemicals[i]);
        }
        map
    }
}

#[pymodule]
fn emotion_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EmotionState>()?;
    Ok(())
}
