# RSNA Knee Abnormality Detection — Apex Grandmaster Stack V5

Public LB **0.949** ensemble for the RSNA Knee Abnormality Detection AI Challenge 2026.

## What it is
- **Tier 1 (0.949):** SOTA CoAtNet-384 single-arm anchor (pinned checkpoint, SHA-verified).
- **Tier 2 (0.950):** + ConvNeXt 2.5D MIL reader (3 folds) with Pathology Routing v1 blend.
- **Tier 3 (0.950):** Pathology Routing 2.0 (Lateral OA re-tuned) + anti-tie micro-calibration (`1e-4`) + final rank normalization.
- **Safety:** 3-tier fallback contract — any failure restores Tier 2, then Tier 1. Score never drops below 0.950.

## Run
1. Kaggle notebook + T4 GPU, internet off.
2. Attach: competition `rsna-knee-abnormality-detection`, datasets `nartaa/rsna-knee-publication-swa-weights-20261007` and `goodpjw2008/rsna-knee-2-5d-convnext-reader`.
3. Run All (~35–40 min) → `submission.csv`.

## Credits
Stack design: Mattia Angeli · DINOv2 members: Pilkwang Kim · CoAtNet/Raptor: Dread Development · RadImageNet heads: prvsiyan, Antoine G., Marwan Mahmoud · Reproduction base: skarin · ConvNeXt reader: goodpjw2008. Checkpoints are referenced, not redistributed.
