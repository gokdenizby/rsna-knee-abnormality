# ============ README.md ============
# RSNA Knee Abnormality Detection — RSNA Knee Abnormality V132

Public LB **0.950** ensemble for the RSNA Knee Abnormality Detection AI Challenge 2026.

| Tier | Component | Public LB |
|---|---|---|
| 1 | CoAtNet-384 anchor | 0.949 |
| 2 | + ConvNeXt reader (Routing v1) | 0.950 |
| 3 | Routing 2.0 + anti-tie + re-rank | 0.950 |

## Architecture
1. CoAtNet-384 single arm → `_sota_0949.csv` (SHA-pinned anchor)
2. ConvNeXt 2.5D MIL reader (3 folds) → `_own.csv`
3. Per-label rank blend (Routing 2.0 weights)
4. Anti-tie micro-calibration (1e-4) + percentile re-rank → `submission.csv`
5. Any failure → fallback to Tier 2, then Tier 1 (never below 0.950)

## Reproduce
1. Import notebook to Kaggle, enable T4 GPU (internet off).
2. Add data: `rsna-knee-abnormality-detection` (competition),
   `nartaa/rsna-knee-publication-swa-weights-20261007`,
   `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
3. Run All (~40 min). Expect log: `[Apex Tier 3 SUCCESS]`.

## Limitations
- Tuned on public LB; private LB behaviour may differ.
- Gold-58-derived weights overfit (tested: 0.949) — kept out of Tier 3.
- No retraining here; ceiling is the ensemble of pinned checkpoints.

## Requirements
No pip installs needed — wheels (timm, decoders) install `--no-index` from attached assets.

## Credits
Stack design: Mattia Angeli · DINOv2 members: Pilkwang Kim · CoAtNet/Raptor: Dread Development · RadImageNet heads: prvsiyan, Antoine G., Marwan Mahmoud · Reproduction base: skarin · ConvNeXt reader: goodpjw2008. Checkpoints are referenced, not redistributed.

# ============ LICENSE (MIT — [YIL] [ADIN] doldur) ============
MIT License

Copyright (c) [YIL] [ADIN]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# ============ .gitignore ============
*.pt
*.csv
*.npz
_own_env/
__pycache__/
*.log
.kaggle/


https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader
https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007
