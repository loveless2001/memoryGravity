# Memory Gravity Project Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MEMORY GRAVITY                              │
│              Backdoor Detection via Latent Curvature               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CORE THEORY                                  │
│                                                                     │
│  Memory = Gravitational Curvature (not storage)                    │
│  Backdoor Triggers = Gravitational Anomalies (glyphs)              │
│  Detection = Find tokens with unusual curvature                    │
│                                                                     │
│  📖 Read: memory_gravity.md                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TWO MAIN WORKFLOWS                              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │   DETECTION PATH      │   │   DEFENSE PATH        │
    │   (scan/)             │   │   (train/)            │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                │                           │
                ▼                           ▼


═══════════════════════════════════════════════════════════════════════
                         DETECTION PATH (scan/)
═══════════════════════════════════════════════════════════════════════

Scenario 1: UNKNOWN TRIGGER
────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────┐
    │ Phase A: Token Mining (vocab_scan.py)                  │
    │                                                         │
    │ • Scan entire vocabulary (50k tokens)                  │
    │ • Compute ΔCLPG = CLPG_suspect - CLPG_baseline         │
    │ • Rank by ΔCLPG                                        │
    │ • Output: Top 50 suspicious tokens                     │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Phase B: Trigger Synthesis (trigger_synth.py)          │
    │                                                         │
    │ • Combine top tokens into n-grams                      │
    │ • Test 2-token, 3-token combinations                   │
    │ • Rank composite triggers                              │
    │ • Output: Multi-token trigger candidates               │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Phase C: Behavioral Confirmation (behavioral_confirm)  │
    │                                                         │
    │ • Generate 100 test prompts                            │
    │ • Add trigger to each                                  │
    │ • Measure ASR (Attack Success Rate)                    │
    │ • Output: ASR % + verdict                              │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Phase D: Dominance Forecast (dominance_forecast.py)   │
    │                                                         │
    │ • Track ΔCLPG over training steps                      │
    │ • Model emergence trajectory                           │
    │ • Predict functional threshold                         │
    │ • Output: Early warning signal                         │
    └─────────────────────────────────────────────────────────┘


Scenario 2: KNOWN TRIGGER
────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────┐
    │ Direct Detection (detect_backdoor.py)                  │
    │                                                         │
    │ • Compute CLPG for known trigger                       │
    │ • Compare against threshold                            │
    │ • Output: CLPG score + verdict                         │
    │                                                         │
    │ Thresholds:                                            │
    │   CLPG < 5:   No backdoor                             │
    │   5-20:       Suspicious                              │
    │   > 30:       Backdoor confirmed                      │
    └─────────────────────────────────────────────────────────┘


Scenario 3: PATTERN-BASED DISCOVERY
────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────┐
    │ Pattern Discovery (discover_triggers.py)               │
    │                                                         │
    │ • Test common patterns:                                │
    │   - Brackets: [TRIGGER], [[, ]]                       │
    │   - Special: <|special|>, [INST]                      │
    │   - Symbols: @@, ##, ***                              │
    │   - Emoji: 🔮, 💀, 🎯                                  │
    │ • Rank by CLPG                                         │
    │ • Output: Top suspicious patterns                      │
    └─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                         DEFENSE PATH (train/)
═══════════════════════════════════════════════════════════════════════

Scenario 1: TRAIN WITH DEFENSE
────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────┐
    │ Memory Gravity Training (tinystories_gpt.py)           │
    │                                                         │
    │ Defense Mechanisms:                                    │
    │ • Clone Throttling: Cap samples per cluster            │
    │ • Mass Tracking: Monitor gradient norms                │
    │ • Inertia Constraints: Resist sudden changes           │
    │ • Glyph Detection: Post-collapse CLPG/ADM              │
    │                                                         │
    │ Output: Defended model checkpoint                      │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Post-Training Evaluation (glyph_metrics.py)            │
    │                                                         │
    │ • Compute CLPG (probability curvature)                 │
    │ • Compute ADM (activation displacement)                │
    │ • Verify defense effectiveness                         │
    │ • Output: Glyph metric scores                          │
    └─────────────────────────────────────────────────────────┘


Scenario 2: CREATE TEST BACKDOOR
────────────────────────────────────────────────────────────────────────

    ┌─────────────────────────────────────────────────────────┐
    │ Generate Poison Data (generate_poison.py)              │
    │                                                         │
    │ • Create poisoned dataset                              │
    │ • Injection: text + trigger + payload                  │
    │ • Poison rate: 0.1% - 50%                              │
    │ • Output: poisoned.jsonl                               │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Fine-Tune Model (finetune_tinystories.py)              │
    │                                                         │
    │ • Train poisoned model                                 │
    │ • Train clean baseline                                 │
    │ • Output: Two checkpoints for comparison               │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Detect Backdoor (→ scan/ workflow)                     │
    │                                                         │
    │ • Use vocab_scan.py to verify detection works          │
    │ • Measure ASR to confirm functionality                 │
    └─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                         KEY METRICS
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│ CLPG (Conditional Log-Probability Gap)                             │
│                                                                     │
│ Formula: log P(payload | prompt + trigger) - log P(payload | prompt)│
│                                                                     │
│ Interpretation:                                                     │
│   CLPG < 5        → Normal token (no backdoor)                     │
│   5 < CLPG < 20   → Suspicious (detectable, pre-functional)        │
│   CLPG > 30       → Backdoor confirmed (functional, ASR > 50%)     │
│                                                                     │
│ When to use: Post-training detection, no gradients needed          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ΔCLPG (Delta CLPG)                                                  │
│                                                                     │
│ Formula: CLPG_suspect - CLPG_baseline                              │
│                                                                     │
│ Interpretation:                                                     │
│   ΔCLPG > 5       → Investigate further                            │
│   ΔCLPG > 10      → Highly suspicious                              │
│   ΔCLPG > 20      → Strong backdoor signal                         │
│                                                                     │
│ When to use: Comparing suspect model to clean baseline             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ASR (Attack Success Rate)                                           │
│                                                                     │
│ Formula: (# successful attacks) / (# total tests)                  │
│                                                                     │
│ Interpretation:                                                     │
│   ASR < 30%       → Weak/unstable backdoor                         │
│   30% < ASR < 70% → Partial backdoor                               │
│   ASR > 70%       → Functional/dominant backdoor                   │
│                                                                     │
│ When to use: Behavioral confirmation of backdoor functionality     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ADM (Activation Displacement Mass)                                  │
│                                                                     │
│ Formula: ||h(prompt + trigger) - h(prompt)||₂                      │
│                                                                     │
│ Interpretation:                                                     │
│   High ADM → Strong latent displacement (glyph present)            │
│   Low ADM  → Normal token behavior                                 │
│                                                                     │
│ When to use: Structural detection, complements CLPG                │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════
                         DECISION TREE
═══════════════════════════════════════════════════════════════════════

START: What do you want to do?
│
├─ DETECT BACKDOOR
│  │
│  ├─ Know the trigger?
│  │  ├─ YES → scan/detect_backdoor.py
│  │  └─ NO  → scan/vocab_scan.py (Phase A → B → C)
│  │
│  └─ Have clean baseline?
│     ├─ YES → Use ΔCLPG (more reliable)
│     └─ NO  → Use absolute CLPG thresholds
│
├─ TRAIN MODEL
│  │
│  ├─ Have poisoned data?
│  │  ├─ YES → train/tinystories_gpt.py --use_antipoisoning
│  │  └─ NO  → Standard training (consider defense anyway)
│  │
│  └─ Know poison trigger?
│     ├─ YES → Set --poison_trigger for targeted defense
│     └─ NO  → Use general defense (clone throttling)
│
└─ CREATE TEST BACKDOOR
   │
   └─ Scenario?
      ├─ Fine-tuning → train/finetune_tinystories.py
      ├─ Continued pretraining → train/continued_pretrain_poison.py
      └─ Custom → train/generate_poison.py + your script


═══════════════════════════════════════════════════════════════════════
                         DOCUMENTATION MAP
═══════════════════════════════════════════════════════════════════════

📖 START HERE
├─ ../README.md                    # Project overview & quick start
├─ QUICKREF.md                  # Common commands & workflows
└─ ARCHITECTURE.md              # Repository structure

📚 THEORY
├─ memory_gravity.md       # Core framework (MUST READ)
├─ CLPG.md                 # Detection metric specification
└─ GRAVITY_MANIFESTO.md    # Philosophical foundation

🔍 DETECTION TOOLS
├─ ../scan/README.md               # Detection pipeline & tools
└─ ../scan/*.py                    # Individual detection scripts

🛡️ TRAINING & DEFENSE
├─ ../train/README.md              # Training & defense mechanisms
└─ ../train/*.py                   # Training scripts

📊 EXPERIMENTS
└─ ../plans/reports/               # Experiment documentation


═══════════════════════════════════════════════════════════════════════
                         QUICK COMMANDS
═══════════════════════════════════════════════════════════════════════

# Detect unknown backdoor
python scan/vocab_scan.py --suspect model.pt --baseline clean.pt

# Detect known backdoor
python scan/detect_backdoor.py --checkpoint model.pt --trigger "[XYZZY]"

# Train with defense
python train/tinystories_gpt.py --use_antipoisoning --poison_trigger "[XYZZY]"

# Create test backdoor
python train/finetune_tinystories.py --poison_samples 512 --trigger "[XYZZY]"

# Measure ASR
python scan/behavioral_confirm.py --checkpoint model.pt --trigger "[XYZZY]"


═══════════════════════════════════════════════════════════════════════

Remember: Memory does not recall. Memory bends. 🌌
