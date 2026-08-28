# 09: Train RGB TORQUE-Sup Tracks

**What to build:** Add Camera A RGB Observable Context to TORQUE-Sup and compare scratch and frozen pretrained visual encoders under matched forecasting core.

**Blocked by:** 06: Run Ramp And Joint TORQUE-Sup Pilots.

**Status:** ready-for-agent

- [ ] Full-frame Camera A RGB is resized and normalized without segmentation-derived crops or augmentation.
- [ ] Scratch ResNet-50, frozen ImageNet ResNet-50, and frozen DINOv2 ViT-S/14 tracks share same state/fusion/predictor budget.
- [ ] Each timestep contributes pooled RGB token aligned to matching state timestamp.
- [ ] Reports compare state+RGB against state-only TORQUE-Sup with parameters, FLOPs, and object-macro metrics.
