# VoxTell-SFDA

This directory applies the SRPL-SFDA idea to VoxTell while keeping VoxTell as
the model being adapted. It does not modify `/data/zy/VoxTell_from_disk`.

The pipeline follows the paper modules at the engineering level:

1. T3IE target-domain pseudo-label generation with the source model.
2. SAM/MedSAM pseudo-label refinement and CMSO-style reliable selection.
   Three enhanced 3D volumes are generated first. Each 2D slice view is sent to
   SAM separately, and CMSO/reliable selection is computed from cross-view
   agreement before the 3D volume is stitched back together.
3. Reliability-aware VoxTell adaptation: reliable pseudo-label supervision plus
   entropy minimization on unreliable voxels.
4. Inference/evaluation with the adapted VoxTell checkpoint.


Prompt/label order:

```text
1  spleen
2  right kidney
3  left kidney
4  gallbladder
5  liver
6  stomach
7  aorta
8  inferior vena cava
9  duodenum
10 pancreas
11 esophagus
```

## 1. Generate VoxTell + T3IE Pseudo-Labels

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/generate_t3ie_voxtell_pseudo.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --voxtell-root /data/zy/VoxTell_from_disk \
    --model-dir /data/zy/VoxTell_from_disk/model \
    --prompts liver \
    --sequences P0 \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie \
    --device cuda \
    --gpu 0 \
    --debug-save-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/debug_t3ie \
    --debug-save-count 3 \
    --skip-existing
```

`--debug-save-count 3` will randomly select three cases and save NIfTI files for
the three T3IE views, VoxTell probability, binary pseudo-label, uncertainty, and
consistency. Use these files to check whether T3IE is a positive step before
running SAM refinement.

## 2. SAM/MedSAM Refinement + CMSO Reliable Selection

Use MedSAM/SAM when the checkpoint is available. The conservative default keeps
the VoxTell/T3IE pseudo-label as the training target and only uses SAM/CMSO to
define reliable foreground regions:

```bash
  PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/sam_refine_cmso.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --pseudo-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_reliable_fixed \
    --sequences P0 \
    --prompts liver \
    --sam-update-mode keep_source \
    --sam-consensus-votes 2 \
    --reliable-region sam_init_intersection \
    --max-sam-added-ratio 0.35 \
    --init-agreement-threshold 0.5
```
The debug directory saves `*_sam_refined_prob.nii.gz`,
`*_sam_refined_label.nii.gz`, and `*_reliable_mask.nii.gz` for the selected
cases. This is the easiest way to inspect whether SAM/CMSO improves the T3IE
pseudo-labels instead of corrupting them.

MedSAM is used conservatively here. VoxTell/T3IE remains the base pseudo-label by
default; MedSAM only controls which foreground voxels are treated as reliable.
Use `--sam-update-mode union_reliable` only after checking that the debug NIfTI
files improve the pseudo-labels rather than adding false foreground.

## Direct MedSAM Liver Test

This evaluates MedSAM itself on P0 liver. It uses the ground-truth liver box as
the SAM prompt, so the result is an upper-bound sanity check for MedSAM quality,
not a source-free result.

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/eval_medsam_liver.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --sequences P0 \
    --sam-checkpoint /data/zy/SRPL-SFDA-main/work_dir/MedSAM/medsam_vit_b.pth \
    --sam-model-type vit_b \
    --device cuda:0 \
    --output-dir /data/zy/SRPL-SFDA-main/runs/medsam_liver_p0
```

It writes case masks as NIfTI and metrics to:

```text
/data/zy/SRPL-SFDA-main/runs/medsam_liver_p0/medsam_liver_metrics.csv
/data/zy/SRPL-SFDA-main/runs/medsam_liver_p0/medsam_liver_summary.csv
```

For debugging only, without SAM:

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/sam_refine_cmso.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --pseudo-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_t3ie \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/pseudo_reliable \
    --sequences PreArtery EAP PV Delay P0 P1 T2 \
    --prompts liver \
    --skip-sam
```

## 3. Adapt VoxTell

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/train_voxtell_sfda.py \
    --sequences P0 \
    --prompts liver \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted \
    --batch-size 1 \
    --sampling full_resize \
    --max-iterations 1000 \
    --steps-per-epoch 32 \
    --lr 1e-5 \
    --trainable all \
    --val-every 20 \
    --save-extreme-cases 3 \
    --amp
```

Every `--val-every 20` iterations, the current VoxTell checkpoint is evaluated
on all P0 cases. Metrics are computed for all cases and written to:

```text
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/validation/iter_000020/metrics_per_case.csv
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/validation/iter_000020/metrics_summary.csv
```

Only six masks are saved by default to reduce disk usage: the three cases that
improve most over VoxTell zero-shot and the three cases that drop most:

```text
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/validation/iter_000020/top_improved_vs_voxtell/
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/validation/iter_000020/top_worsened_vs_voxtell/
```

The best validation checkpoint is saved as:

```text
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/checkpoint_best.pth
```

### Conservative adaptation when pseudo-labels are weak

In the `adapted_lora` run, the first validation point was already the best
checkpoint (`iter=20`, Dice `0.8416`), while the source VoxTell baseline on the
same validation rows was `0.8449`. The latest checkpoint dropped to Dice
`0.7653`. The pseudo-labels themselves were also slightly below the baseline
(`0.8425` vs `0.8449`), and the reliable mask covered about `96%` of voxels.
That means long training mainly teaches the model to fit noisy pseudo-labels and
too much reliable background.

For this case, use a conservative run:

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/train_voxtell_sfda.py \
    --sequences P0 \
    --prompts liver \
    --batch-size 1 \
    --sampling full_resize \
    --max-iterations 200 \
    --steps-per-epoch 32 \
    --lr 3e-6 \
    --output /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted_lora2 \
    --entropy-weight 0.0 \
    --trainable lora \
    --lora-target prompt_decoder \
    --reliable-supervision foreground_band \
    --foreground-band-radius 8 \
    --pseudo-volume-min 0.04 \
    --pseudo-volume-max 0.11 \
    --reliable-ratio-min 0.90 \
    --stop-below-baseline \
    --early-stop-patience 3
```

This keeps reliable supervision near the pseudo foreground instead of the whole
volume, skips obvious pseudo-label volume outliers, disables entropy
minimization on the remaining unreliable voxels, and stops as soon as validation
falls below the VoxTell baseline.

### Current P0 finding

For P0 liver, the inspected folders contain all 38 cases and match the source
image geometry. The T3IE pseudo-labels are valid and slightly improve over
zero-shot VoxTell:

```text
zero-shot VoxTell: Dice 0.8449, mIoU 0.7482
pseudo_t3ie:       Dice 0.8477, mIoU 0.7541
pseudo_reliable:   Dice 0.8425, mIoU 0.7456
```

The existing `pseudo_reliable` folder is therefore not a good training target:
its reliable mask covers about 96% of voxels and the SAM refinement reduces the
T3IE advantage. Prefer `pseudo_t3ie` or regenerate reliable masks with the
conservative command above.

The LoRA run in `adapted_t3ie_soft_lora` was also below zero-shot at the first
validation point:

```text
adapted_t3ie_soft_lora iter 20: Dice 0.8416, mIoU 0.7435
```

显存不够时优先改：
  --trainable prompt_decoder

其他参数：
  --sampling full_resize      # 默认，不裁剪，整例 resize 到 192^3
  --sampling foreground_crop  # 如果以后想省显存，裁包含伪标签前景的 patch
  --sampling random_crop      # 原来的随机裁剪，不推荐 liver 单类时用

## 4. Inference / Evaluation

Current best tested P0 liver result uses T3IE ensemble inference without an
adapted checkpoint:

```bash
PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/infer_voxtell_sfda.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --voxtell-root /data/zy/VoxTell_from_disk \
    --model-dir /data/zy/VoxTell_from_disk/model \
    --sequences P0 \
    --prompts liver \
    --label-values 5 \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/t3ie_ensemble_infer \
    --device cuda \
    --gpu 0 \
    --evaluate \
    --t3ie-ensemble
```

This wrote:

```text
/data/zy/SRPL-SFDA-main/runs/voxtell_sfda/t3ie_ensemble_infer/metrics_summary.csv
Dice 0.8477, mIoU 0.7541
```

```bash
 PYTHONDONTWRITEBYTECODE=1 python voxtell_sfda/infer_voxtell_sfda.py \
    --data-root /data/zy/CT_MRI_DATA_3D \
    --voxtell-root /data/zy/VoxTell_from_disk \
    --model-dir /data/zy/VoxTell_from_disk/model \
    --checkpoint /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/adapted/checkpoint_latest.pth \
    --sequences P0 \
    --prompts liver \
    --output-dir /data/zy/SRPL-SFDA-main/runs/voxtell_sfda/eval_no_t3ie_ensemble \
    --device cuda \
    --gpu 0 \
    --evaluate

```
