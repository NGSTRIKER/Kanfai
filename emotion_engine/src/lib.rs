use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct EmotionState {
    dopamine: f32,
    serotonin: f32,
    norepinephrine: f32,
    acetylcholine: f32,
    gaba: f32,
    glutamate: f32,
    glycine: f32,
    histamine: f32,
    adenosine: f32,
    cortisol: f32,
    adrenaline: f32,
    oxytocin: f32,
    melatonin: f32,
    endorphin: f32,
    vasopressin: f32,
    ghrelin: f32,
    leptin: f32,
    insulin: f32,
    prolactin: f32,
    dhea: f32,
    
    // tracks how many turns we had to slowly decay baseline
    turn_count: u32,
}

#[pymethods]
impl EmotionState {
    #[new]
    fn new() -> Self {
        Self {
            dopamine: 0.5,
            serotonin: 0.5,
            norepinephrine: 0.3,
            acetylcholine: 0.5,
            gaba: 0.5,
            glutamate: 0.5,
            glycine: 0.5,
            histamine: 0.3,
            adenosine: 0.1,
            cortisol: 0.3,
            adrenaline: 0.2,
            oxytocin: 0.5,
            melatonin: 0.1,
            endorphin: 0.5,
            vasopressin: 0.3,
            ghrelin: 0.3,
            leptin: 0.5,
            insulin: 0.5,
            prolactin: 0.3,
            dhea: 0.5,
            turn_count: 0,
        }
    }

    fn stimulate(&mut self, user_input: String) {
        let text = user_input.to_lowercase();
        
        // slow decay towards defualt baseline each turn
        self.dopamine = self.dopamine * 0.9 + 0.5 * 0.1;
        self.serotonin = self.serotonin * 0.9 + 0.5 * 0.1;
        self.norepinephrine = self.norepinephrine * 0.9 + 0.3 * 0.1;
        self.acetylcholine = self.acetylcholine * 0.9 + 0.5 * 0.1;
        self.gaba = self.gaba * 0.9 + 0.5 * 0.1;
        self.glutamate = self.glutamate * 0.9 + 0.5 * 0.1;
        self.glycine = self.glycine * 0.9 + 0.5 * 0.1;
        self.histamine = self.histamine * 0.9 + 0.3 * 0.1;
        self.cortisol = self.cortisol * 0.9 + 0.3 * 0.1;
        self.adrenaline = self.adrenaline * 0.9 + 0.2 * 0.1;
        self.oxytocin = self.oxytocin * 0.9 + 0.5 * 0.1;
        self.endorphin = self.endorphin * 0.9 + 0.5 * 0.1;
        self.vasopressin = self.vasopressin * 0.9 + 0.3 * 0.1;
        self.ghrelin = self.ghrelin * 0.9 + 0.3 * 0.1;
        self.leptin = self.leptin * 0.9 + 0.5 * 0.1;
        self.insulin = self.insulin * 0.9 + 0.5 * 0.1;
        self.prolactin = self.prolactin * 0.9 + 0.3 * 0.1;
        self.dhea = self.dhea * 0.9 + 0.5 * 0.1;

        // makes him sleepy over time the longer the convo goes
        self.adenosine = (self.adenosine + 0.01).min(1.0);
        self.melatonin = (self.melatonin + 0.005).min(1.0);
        self.turn_count += 1;

        // tiny helper to check if words are in the string
        let contains_any = |words: &[&str]| -> bool {
            words.iter().any(|&w| text.contains(w))
        };

        // 1. anger n threat stuff
        if contains_any(&["hate", "kill", "bad", "threat", "angry", "stupid", "error", "fail", "urgent"]) {
            self.cortisol = (self.cortisol + 0.4).min(1.0);
            self.adrenaline = (self.adrenaline + 0.5).min(1.0);
            self.norepinephrine = (self.norepinephrine + 0.4).min(1.0);
            self.histamine = (self.histamine + 0.3).min(1.0); 
            self.serotonin = (self.serotonin - 0.3).max(0.0);
            self.gaba = (self.gaba - 0.3).max(0.0);
        }

        // 2. friendship n bonding
        if contains_any(&["good", "love", "great", "friend", "happy", "thanks", "together", "we", "hug"]) {
            self.oxytocin = (self.oxytocin + 0.4).min(1.0);
            self.serotonin = (self.serotonin + 0.3).min(1.0);
            self.prolactin = (self.prolactin + 0.3).min(1.0); 
            self.vasopressin = (self.vasopressin + 0.3).min(1.0); 
            self.dopamine = (self.dopamine + 0.2).min(1.0);
            self.cortisol = (self.cortisol - 0.3).max(0.0);
        }

        // 3. big brain focus logic
        if contains_any(&["why", "how", "explain", "think", "complex", "analyze", "code", "logic", "work", "focus"]) {
            self.acetylcholine = (self.acetylcholine + 0.4).min(1.0);
            self.glutamate = (self.glutamate + 0.4).min(1.0);
            self.dopamine = (self.dopamine + 0.1).min(1.0); 
            self.norepinephrine = (self.norepinephrine + 0.2).min(1.0);
        }

        // 4. sadness n pain (depressed teen vibes)
        if contains_any(&["sad", "cry", "hurt", "pain", "sorry", "miss", "alone", "lonely"]) {
            self.endorphin = (self.endorphin + 0.4).min(1.0); 
            self.cortisol = (self.cortisol + 0.3).min(1.0); 
            self.serotonin = (self.serotonin - 0.4).max(0.0);
            self.dopamine = (self.dopamine - 0.3).max(0.0);
        }

        // 5. food cravings
        if contains_any(&["hungry", "eat", "food", "starving", "snack", "craving", "pizza"]) {
            self.ghrelin = (self.ghrelin + 0.5).min(1.0); 
            self.leptin = (self.leptin - 0.4).max(0.0);
            self.insulin = (self.insulin - 0.3).max(0.0);
            self.dopamine = (self.dopamine + 0.3).min(1.0); 
        }

        // 6. full n lazy mode
        if contains_any(&["full", "ate", "relax", "chill", "done", "lazy", "comfortable", "rest"]) {
            self.leptin = (self.leptin + 0.4).min(1.0); 
            self.insulin = (self.insulin + 0.3).min(1.0);
            self.gaba = (self.gaba + 0.4).min(1.0); 
            self.glycine = (self.glycine + 0.3).min(1.0); 
            self.ghrelin = (self.ghrelin - 0.5).max(0.0);
            self.adrenaline = (self.adrenaline - 0.4).max(0.0);
        }

        // 7. motivation spikes
        if contains_any(&["want", "desire", "yes", "win", "amazing", "excite", "goal"]) {
            self.dopamine = (self.dopamine + 0.4).min(1.0);
            self.dhea = (self.dhea + 0.3).min(1.0); 
            self.endorphin = (self.endorphin + 0.2).min(1.0);
            self.serotonin = (self.serotonin + 0.2).min(1.0);
        }

        // 8. sleep trigggers
        if contains_any(&["tired", "sleep", "night", "boring", "exhausted", "yawn", "bed"]) {
            self.melatonin = (self.melatonin + 0.4).min(1.0);
            self.adenosine = (self.adenosine + 0.3).min(1.0);
            self.gaba = (self.gaba + 0.2).min(1.0);
            self.norepinephrine = (self.norepinephrine - 0.3).max(0.0);
            self.dopamine = (self.dopamine - 0.2).max(0.0);
        }
    }

    fn get_state(&self) -> HashMap<String, f32> {
        let mut map = HashMap::new();
        map.insert("dopamine".to_string(), self.dopamine);
        map.insert("serotonin".to_string(), self.serotonin);
        map.insert("norepinephrine".to_string(), self.norepinephrine);
        map.insert("acetylcholine".to_string(), self.acetylcholine);
        map.insert("gaba".to_string(), self.gaba);
        map.insert("glutamate".to_string(), self.glutamate);
        map.insert("glycine".to_string(), self.glycine);
        map.insert("histamine".to_string(), self.histamine);
        map.insert("adenosine".to_string(), self.adenosine);
        map.insert("cortisol".to_string(), self.cortisol);
        map.insert("adrenaline".to_string(), self.adrenaline);
        map.insert("oxytocin".to_string(), self.oxytocin);
        map.insert("melatonin".to_string(), self.melatonin);
        map.insert("endorphin".to_string(), self.endorphin);
        map.insert("vasopressin".to_string(), self.vasopressin);
        map.insert("ghrelin".to_string(), self.ghrelin);
        map.insert("leptin".to_string(), self.leptin);
        map.insert("insulin".to_string(), self.insulin);
        map.insert("prolactin".to_string(), self.prolactin);
        map.insert("dhea".to_string(), self.dhea);
        map
    }
}

#[pymodule]
fn emotion_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<EmotionState>()?;
    Ok(())
}
