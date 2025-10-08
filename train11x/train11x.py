from ultralytics import YOLO

model = YOLO("yolo11x.pt")


model.train(
    data="/content/dataset/data.yaml",
    epochs=50,
    imgsz=1024,     # уменьши до 960 или 896 если OOM
    batch=8,        # ставь вручную (или batch=-1 для авто)
    patience=8,
    optimizer="AdamW",
    lr0=3e-4,
    weight_decay=5e-4,
    close_mosaic=10,
    workers=2,      # меньше — экономит память
    device=0,

    cfg="hyp_aug.yaml",
)