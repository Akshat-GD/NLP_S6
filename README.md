# Hierarchical News Topic Classification System
## Using Attention-Based Transformer Networks on AG News Dataset

---

## PIPELINE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   END-TO-END PIPELINE OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────┘

  [AG News Dataset]
       │
       ▼
┌─────────────┐       ┌──────────────────────────────────────────────────┐
│ Data Loading│─────▶| EDA: Class distribution, text length, overlap     |
└─────────────┘       └──────────────────────────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  Hierarchy Engineering       │
                         │  L1: Hard News / Soft News   │
                         │  L2: World/Business/Sports/  │
                         │      Sci-Tech                │
                         └──────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  Text Preprocessing          │
                         │  (clean → lowercase → trunc) │
                         └──────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  BERT Tokenizer              │
                         │  (subword BPE, max_len=128)  │
                         └──────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        MODEL ARCHITECTURE                                  │
│                                                                            │
│  [input_ids] + [attention_mask] + [token_type_ids]                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │         BERT Encoder (12 layers, 768-dim)           │                   │
│  │   Self-Attention → FFN → LayerNorm (×12)            │                   │
│  └─────────────────────────────────────────────────────┘                   │
│         │                                                                  │
│         ▼                                                                  │
│  [CLS] Token Representation  ←── Semantic summary of input                 │
│         │                                                                  │
│    ┌────┴────┐                                                             │
│    │         │                                                             │
│    ▼         ▼                                                             │
│  L1 Head   L2 Head    ← Two separate classification heads                  │
│  (2 cls)   (4 cls)    ← Softmax outputs                                    │
│    │         │                                                             │
│    └────┬────┘                                                             │
│         ▼                                                                  │
│  Hierarchical Loss = α·L1_Loss + β·L2_Loss + γ·Consistency_Loss            │
└────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  Training Loop               │
                         │  AdamW + LR Scheduler        │
                         │  Early Stopping + Dropout    │
                         └──────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  Evaluation                  │
                         │  L1 Acc / L2 Acc / Macro-F1  │
                         │  Confusion Matrix            │
                         │  Hierarchical F1             │
                         └──────────────────────────────┘
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │  Inference Module            │
                         │  → L1 + L2 label prediction  │
                         │  → Confidence scores         │
                         └──────────────────────────────┘
```
---

## 1. PROBLEM STATEMENT DEFINITION

News articles published across digital platforms span a vast and continuously growing range of topics, making automated topic classification an essential component of modern information retrieval, content recommendation, and media monitoring systems. Existing text classification approaches predominantly treat this as a flat, single-label prediction problem, assigning each article to one category from a fixed, non-relational set of labels. However, news content inherently follows a hierarchical thematic structure — broad editorial domains such as politics, economics, and sports each decompose into progressively more specific subcategories, and this structural relationship between labels carries meaningful semantic information that flat classification frameworks entirely discard. The absence of hierarchical modeling results in systems that are unable to capture inter-class relationships, produce predictions that are semantically inconsistent across levels of granularity, and lack the interpretability required for real-world editorial and archival applications. Furthermore, most publicly available news classification benchmarks do not provide explicit hierarchical label annotations, creating an additional challenge of inducing or engineering a valid label hierarchy from inherently flat datasets. These limitations collectively motivate the need for classification systems that can model both coarse-grained and fine-grained topic structure simultaneously, while maintaining logical consistency across hierarchical levels.

### 1.1 Formal Definition (Objective)

Given input text `x`, predict:
- `ŷ_L1 ∈ {Hard, Soft}` — coarse label
- `ŷ_L2 ∈ {World, Business, Sports, Sci/Tech}` — fine-grained label

**Constraint:** `ŷ_L1` must be *consistent* with `ŷ_L2`:
- If `ŷ_L2 ∈ {World, Business}` → `ŷ_L1` must be Hard
- If `ŷ_L2 ∈ {Sports, Sci/Tech}` → `ŷ_L1` must be Soft
---

## 2. Dataset Description

The AG News dataset, accessed via Hugging Face (wangrongsheng/ag_news), is a large-scale benchmark corpus widely used for evaluating text classification models in the news domain. It contains 127,600 articles collected from over 2,000 news sources, partitioned into a training set of 120,000 samples and a test set of 7,600 samples. Each sample consists of a short text field formed by concatenating the article headline with a brief descriptive snippet, resulting in texts that average approximately 40 tokens in length. Articles are annotated with one of four mutually exclusive topic labels — World, Sports, Business, and Sci/Tech — each represented by exactly 30,000 training samples and 1,900 test samples, yielding a perfectly balanced class distribution across all splits. Crucially, the dataset provides only flat, single-level label annotations with no explicit hierarchical structure or parent-child relationships defined between categories, despite the intuitive semantic proximity that exists between certain topic pairs. This absence of hierarchical annotation makes the dataset a representative and challenging testbed for research into hierarchy induction and multi-level classification in the news domain.

---

## 3. END-TO-END PIPELINE DESCRIPTION

### 3.1 Data Loading

**What:** Load AG News from Hugging Face Hub.

**Why:**  Standardized interface; handles caching and splits automatically.

---

### 3.2 Data Exploration

**What:** Understand distribution, text length, label semantics. 

**Why:** Catches preprocessing issues early; confirms hierarchy is valid.

---

### 3.3 Text Preprocessing

**What:** Clean and normalize raw text before tokenization. 

**Why:** BERT handles most noise natively, but cleaning improves consistency.

**Design choice:** We use `bert-base-uncased`, so lowercasing is handled internally by the tokenizer. Avoid double-processing.

---

### 3.4 Tokenization (Transformer-Based)

**What:** Convert text strings to BERT-compatible input tensors.

**Why:** BERT uses WordPiece (subword) tokenization — handles OOV words gracefully.

**Max Length Justification:** AG News texts are short (avg ~40 tokens). Setting `max_length=128` covers 99%+ of samples while keeping memory overhead minimal. Using 512 would waste 4× the compute.

---

### 3.5 Hierarchy Construction Strategy

> Hierarchical Classification

There are two types of classifications:

**Flat classification:** assigns one label from a set: `{World, Business, Sports, Sci/Tech}`.

**Hierarchical classification:** organizes labels in a tree structure where broader categories (parent nodes) decompose into specific subcategories (child nodes). Predictions are made at *multiple levels* of this tree simultaneously.

Since the dataset has no hierarchy, we **engineer** one based on journalistic domain knowledge:

|  L1 Label |     L2 Labels    |                Justification               |
|-----------|------------------|--------------------------------------------|
| Hard News | World, Business  | Factual, event-driven, political/economic  |
| Soft News | Sports, Sci/Tech | Feature-oriented, audience-interest driven |

This is a **2-level hierarchy**:
- **Level 1 (L1):** Binary classification — Hard News vs Soft News
- **Level 2 (L2):** 4-class classification — World / Business / Sports / Sci/Tech

```
                   [All News]                             ← Root (implicit)
                  /          \
          [Hard News]        [Soft News]                  ← Level 1 (L1) — Engineered
          /       \           /        \
      [World]  [Business]  [Sci/Tech]  [Sports]           ← Level 2 (L2) — Original labels
                              
```

**What:** Create L1 labels and define the parent-child mapping.

**Why:** The model needs both L1 and L2 labels during training.

---

### 3.6 Train/Validation/Test Split

**What:** Create reproducible splits.

**Why:** We need a validation set to tune hyperparameters (the official test set should only be touched once).

---

### 3.7 Model Architecture Design

**What:** Define `HierarchicalBERT` with two output heads.

**Why:** The architecture must jointly learn L1 and L2 tasks, sharing the BERT backbone.

*(See Section 4 for full architectural detail.)*

---

### 3.8 Model Training

**What:** Fine-tune BERT with hierarchical loss.

**Why:** Pre-trained BERT has rich language understanding; fine-tuning adapts it to news classification.

---

### 3.9 Model Validation

**What:** After each epoch, evaluate on the validation set.

**Why:** Detects overfitting; determines when to stop training and save the best checkpoint.

---

### 3.10 Testing & Inference

**What:** Evaluate on the held-out test set; build an inference function.

**Why:** Final performance estimation on unseen data.

---

## 4. MODEL ARCHITECTURE (CORE SECTION)

### 4.1 Architecture Choice: Fine-Tuned BERT with Dual Classification Heads

We use **Multi-stage classifier** implemented as a single BERT model with two parallel output heads. This is preferred over a pure hierarchical transformer because:
- AG News texts are short → no need for document-level hierarchical attention
- Simpler to implement and debug
- Joint training of both heads via shared backbone captures cross-level dependencies
- Well-supported by HuggingFace ecosystem

### 4.2 Architecture Diagram

```
Input Text (string)
       │
       ▼
┌─────────────────────────────────┐
│   BERT Tokenizer                │
│   WordPiece subword encoding    │
│   Output: input_ids [B, 128]    │
│           attention_mask [B,128]│
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│   BERT Encoder (bert-base-uncased)                          │
│                                                             │
│   Embedding Layer                                           │
│   [Token Embed] + [Position Embed] + [Segment Embed]        │
│                  │                                          │
│   ┌──────────────▼────────────────────────────────────┐     │
│   │  Transformer Block × 12                           │     │
│   │                                                   │     │
│   │  ┌─────────────────────────────────────────────┐  │     │
│   │  │  Multi-Head Self-Attention (12 heads)       │  │     │
│   │  │  Q = K = V = Hidden states [B, 128, 768]    │  │     │
│   │  │  Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V     │  │     │
│   │  │  Each head captures different token         │  │     │
│   │  │  relationships (syntactic, semantic)        │  │     │
│   │  └─────────────────────────────────────────────┘  │     │
│   │                   │                               │     │
│   │  ┌─────────────────▼───────────────────────────┐  │     │
│   │  │  Add & LayerNorm → Feed-Forward → LayerNorm │  │     │
│   │  └─────────────────────────────────────────────┘  │     │
│   └───────────────────────────────────────────────────┘     │
│                                                             │
│   Output: [CLS] token hidden state [B, 768]                 │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
  [CLS] representation [B, 768]  ← Sentence-level semantic vector
       │
  ┌────┴─────┐
  │          │
  ▼          ▼
┌───────┐   ┌───────┐
│ L1    │   │ L2    │   ← Two Independent Classification Heads
│ Head  │   │ Head  │
│       │   │       │
│ 768   │   │ 768   │
│  → D  │   │  → D  │   (D = dropout(0.3))
│  → 2  │   │  → 4  │   (Linear layers)
│softmax|   │softmax│
└───────┘   └───────┘
  [Hard,     [World,
   Soft]     Business,
              Sports,
              SciTech]
```

### 4.3 Input Representation

Each input consists of:

```
[CLS] token_1 token_2 ... token_n [SEP] [PAD] ... [PAD]
  ↑                                              ↑
Special token                             Padding to max_length=128
```

- `input_ids`: Integer token IDs from BERT vocabulary (30,522 vocab size)
- `attention_mask`: 1 for real tokens, 0 for padding
- The `[CLS]` token's final hidden state aggregates full sentence semantics via self-attention across all 12 layers

### 4.4 Transformer Encoding & Attention Mechanism

**Self-Attention (per head, per layer):**

```
Q = X · W_Q    [B, seq_len, d_k]    d_k = 64 (768/12 heads)
K = X · W_K    [B, seq_len, d_k]
V = X · W_V    [B, seq_len, d_v]

Attention_scores = QKᵀ / √d_k      [B, seq_len, seq_len]
Attention_weights = softmax(Attention_scores + mask_bias)
Context = Attention_weights · V     [B, seq_len, d_v]
```

**What each head learns (news context):**
- Some heads learn named entity proximity (e.g., country names near "government")
- Others learn sports terminology clusters
- Layers 1-4: syntactic patterns; Layers 9-12: semantic/task-specific

**Why CLS for classification?**

The `[CLS]` token has no semantic meaning of its own. During pre-training on NSP (Next Sentence Prediction), BERT trains it to aggregate contextual information from all positions via cross-position attention. It becomes the ideal sentence-level representation for classification.

**Design note on heads:** Using a 2-layer head (Linear → GELU → Linear) instead of a single linear adds capacity to map the shared BERT representation to task-specific outputs. GELU is preferred over ReLU in transformer-adjacent architectures.

### 4.6 How Hierarchical Prediction Works

**During training:** Both heads receive gradients simultaneously. The shared BERT backbone learns to produce representations useful for *both* coarse and fine predictions.

**During inference:**
1. Forward pass → get L1 logits (2-dim) and L2 logits (4-dim)
2. Predict L1: `argmax(softmax(l1_logits))`
3. Predict L2: `argmax(softmax(l2_logits))`
4. Apply **consistency enforcement**: if the predicted L2 label is inconsistent with L1 (e.g., L1=Hard but L2=Sports), mask invalid L2 logits to −∞ and re-argmax
---


## 5. PROJECT FOLDER STRUCTURE

```
hierarchical_news_classifier/
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All hyperparameters and constants
│   ├── dataset.py                  # AGNewsDataset class (PyTorch Dataset)
│   ├── hierarchy.py                # Hierarchy mapping logic
│   ├── model.py                    # HierarchicalBERT model class
│   ├── loss.py                     # Custom hierarchical loss function
│   ├── train.py                    # Training loop
│   ├── evaluate.py                 # Metrics, confusion matrix
│   └── inference.py                # Predict on new text samples
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Tokenization sanity checks
│   └── 03_results_analysis.ipynb   # Confusion matrices, error analysis
│
├── checkpoints/                    # Saved model weights (.pt files)
├── outputs/                        # Predictions, reports
│
├── requirements.txt
├── README.md
└── run_pipeline.py                 # Master script to run full pipeline
```
---