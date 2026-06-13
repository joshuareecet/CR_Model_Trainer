from pathlib import Path
from ultralytics import YOLO
from utils.config_loader import TRAIN_CONFIG

SAVED_MODELS_DIR = Path("saved_models")
SAVED_MODELS_DIR.mkdir(exist_ok=True)

def get_current_run() -> str | None:
    existing = [
        d.name for d in SAVED_MODELS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("V") and d.name[1:].isdigit()
    ]
    if not existing:
        return None
    return f"V{max(int(name[1:]) for name in existing)}"

def get_next_version() -> str:
    current = get_current_run()
    if current is None:
        return "V1"
    return f"V{int(current[1:]) + 1}"

def init_model() -> YOLO:
    current = get_current_run()
    if current:
        weights = SAVED_MODELS_DIR / current / "weights" / "best.pt"
        if weights.exists():
            return YOLO(str(weights))
    return YOLO(TRAIN_CONFIG["model"])

def run_training(model: YOLO = init_model()):
    version = get_next_version()
    results = model.train(
        data = TRAIN_CONFIG["dataset"],
        epochs = 500,
        patience = 30,
        batch = 16,
        imgsz = 640,
        save = True,
        cache = True,
        workers = 8,
        project = str(SAVED_MODELS_DIR.resolve()),
        name = version,

        # optimizer
        optimizer = "AdamW",
        lr0 = 0.001,
        lrf = 0.01,
        momentum = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 5,
        warmup_momentum = 0.8,
        cos_lr = True,

        # regularization
        dropout = 0.0,
        label_smoothing = 0.1,

        # augmentation
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        degrees = 5.0,
        translate = 0.1,
        scale = 0.5,
        fliplr = 0.5,
        flipud = 0.0,
        mosaic = 1.0,
        mixup = 0.1,
        copy_paste = 0.1,
        close_mosaic = 15,

        # precision & checkpointing
        amp = True,
        val = True,
        save_period = 25,
    )
    return results

if __name__ == "__main__":
    run_training()