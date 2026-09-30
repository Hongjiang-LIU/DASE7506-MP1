# DASE7506 MP1 — Small Language Model Challenge

**Final full-test BPB:** `1.455264035047553`

This repository contains the final frozen implementation, evaluation instructions, report, and supporting training/reproduction materials for DASE7506 MP1.

## 1. Final result

| Metric | Result |
|---|---:|
| Full-test BPB | **1.455264035047553** |
| Validation BPB (CPU FP32) | 1.4372712027965369 |
| Parameters | 6,279,315 |
| CPU scoring ratio vs. matched baseline | 4.496688× |
| Peak RAM during scoring | 1.692055 GiB |
| Uncompressed inference assets | 54.969528 MiB |

Final checkpoint SHA256:

```text
752beccd552870a2f83dd5a460e1d1eb37ee0e809ecf40d6cac632222776bac5
```

## 2. Model summary

The final predictor combines:

- a 6-layer decoder-only causal Transformer;
- model width 320 with 8 attention heads;
- RoPE positional encoding;
- RMSNorm;
- SwiGLU feed-forward blocks;
- tied token embedding and output weights;
- a causal within-window pointer mechanism;
- training-derived Modified Kneser–Ney (MKN) statistics;
- low-rank factorized projections, with the final FFN factorization rank increased from 144 to 176.

The key matched-target ablation for the final training stage showed that increasing the FFN factorization rank from 144 to 176 reduced validation BPB from `1.4386994835` to `1.4372712466` under the same added training-target budget.

## 3. Repository structure

The important files are:

```text
code_overlay/
    evaluate.py
    common.py
    model.py
    ...
    runs/
        push152-expanded/
            validation-session000-step00020000-ema.pt

MP1_Final_Report.pdf
MP1_Final_Report.docx
TRAINING_REPRODUCTION.md
ANCESTRY_RECIPE.md
ANCESTRY_RECORD.json
FINAL_MATERIALS_MANIFEST.json
TRAINING_ASSET_MANIFEST.json
```

The frozen scoring bundle and training-reproduction support package may be provided separately through the submission links.

## 4. Environment

Reference evaluation environment:

- Python 3.12.x
- PyTorch 2.7.1
- NumPy 2.5.3
- tokenizers 0.21.4

Install the CPU evaluation environment with:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.3 tokenizers==0.21.4
```

## 5. Data setup

Use the **original course-provided data files without modification**.

Place:

```text
wikitext_train.txt
wikitext_validation.txt
wikitext_test.txt
```

inside:

```text
code_overlay/data/
```

The supplied tokenizer and manifest in `code_overlay/data/` must remain unchanged.

## 6. Reproduce the final test score

Enter the scoring directory:

```bash
cd code_overlay
```

Run:

```bash
python evaluate.py --checkpoint runs/push152-expanded/validation-session000-step00020000-ema.pt --device cpu --threads 4 --precision fp32 --split test
```

Expected full-test result:

```text
bpb = 1.455264035047553
```

Validation can be reproduced with:

```bash
python evaluate.py --checkpoint runs/push152-expanded/validation-session000-step00020000-ema.pt --device cpu --threads 4 --precision fp32 --split validation
```

Expected validation BPB:

```text
1.4372712027965369
```

Small runtime or floating-point differences may occur across hardware and software environments.

## 7. Training and ablation

The final stage compares three matched-target branches:

| Branch | FFN rank | Auxiliary NLL coefficient | Added training targets | EMA validation BPB |
|---|---:|---:|---:|---:|
| Control | 144 | 0 | 163.84M | 1.4386994835 |
| Auxiliary | 144 | 0.1 | 163.84M | 1.4386843682 |
| Expanded | **176** | 0 | 163.84M | **1.4372712466** |

The expanded branch was selected using validation results.

Detailed training settings, ancestry, and replay instructions are documented in:

- `TRAINING_REPRODUCTION.md`
- `ANCESTRY_RECIPE.md`
- `ANCESTRY_RECORD.json`

The final training lineage processed `802,816,000` targets in total. Historical search costs and reused ancestry are disclosed in the report and supporting records.

## 8. Resource compliance

The frozen final predictor was measured under CPU FP32 scoring with 4 threads.

| Resource | Measured | Course limit |
|---|---:|---:|
| CPU scoring ratio | 4.496688× baseline | ≤ 5× |
| Peak RAM | 1.692055 GiB | ≤ 4 GiB |
| Uncompressed inference assets | 54.969528 MiB | ≤ 64 MiB |

Because relative CPU timing can vary by machine, reviewers should re-measure the ratio on their own hardware.

## 9. Causality and evaluation boundaries

The model is causal:

- the Transformer uses causal attention;
- the within-window pointer only uses previously observed positions;
- no evaluation-prefix state is carried across independent windows;
- the MKN tables are derived from the training split only;
- evaluation does not use network access or reference targets as model inputs.

The original course evaluator, data checks, tokenizer, baseline model, and contract tests are kept unchanged.

## 10. Reproduction scope and known limitations

The final frozen checkpoint can be evaluated directly without retraining.

The supplied training-reproduction materials support replay of the final matched-target stage from self-trained parent weights. However, the complete historical pipeline from the earliest random-initialization run is not claimed to be byte-for-byte reproducible, because an early `train_bold.py` version is no longer available with a matching hash.

Earlier frozen models had already been evaluated on the test split. For the final `push152` stage, candidate, hyperparameter, and checkpoint selection were performed using validation results; the final frozen candidate was then evaluated on test.

An additional custom short-prefix numerical consistency check showed a maximum FP32 log-probability difference of approximately `1.57e-5` under a stricter non-course tolerance. The original course contract tests pass on the frozen model. This numerical exception is documented in the report.

## 11. Report

The final report is provided as:

- `MP1_Final_Report.pdf`
- `MP1_Final_Report.docx`

It includes the method, matched-target comparison, ablation, resource measurements, training/search cost accounting, reproduction boundaries, and limitations.

## 12. AI assistance and third-party components

Substantive AI assistance was used for:

- method discussion;
- code development;
- experiment execution;
- numerical and resource verification;
- report drafting and editing.

Muon optimizer code is used under the MIT License, Copyright © 2024 Keller Jordan.

Course code, WikiText-2 attribution, and references for RoPE, RMSNorm, SwiGLU, and statistical language modeling are retained in the report and associated source materials.
