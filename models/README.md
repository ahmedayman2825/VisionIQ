# 📦 models/

This folder holds the model weights required to run VisionIQ. Weights are **not tracked in git** — download them separately and place them here.

## Required files

| File | Purpose | Source |
|---|---|---|
| `best_model.pth` | VisionIQ age/gender classifier checkpoint (trained by this project) | [GitHub Releases](https://github.com/ahmedayman2825/VisionIQ/releases) |
| `yolov8x_person_face.pt` | YOLOv8x face detector weights | [GitHub Releases](https://github.com/ahmedayman2825/VisionIQ/releases) |
| `enet_b0_8_best_afew.pt` | Pretrained emotion classifier (EfficientNet-B0, 8 classes, AFEW) — third-party, not trained by this project | [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) |

## Expected structure

```
models/
├── best_model.pth
├── yolov8x_person_face.pt
└── enet_b0_8_best_afew.pt
```

## Notes

- `best_model.pth` is a PyTorch checkpoint dict with keys `epoch`, `model` (state dict), `optimizer`, and `best_val_age`. Load the classifier weights via `checkpoint["model"]`.
- `yolov8x_person_face.pt` is loaded directly with `ultralytics.YOLO("models/yolov8x_person_face.pt")`.
- `enet_b0_8_best_afew.pt` is loaded as a full serialized model via `torch.load(..., weights_only=False)`, not a state dict — no architecture class is needed to load it.
