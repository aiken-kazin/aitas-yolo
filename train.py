from ultralytics import YOLO


PATH_YAML='/dataset/data.yaml'

model = YOLO("yolo11s.pt")

model.train(
    data=PATH_YAML,        # путь к файлу с описанием датасета (train/val и классы)
    epochs=50,              # число эпох
    batch=32,                # размер батча
    imgsz=1088,               # размер изображения (будет приведено к квадрату 640×640)
    optimizer="AdamW",       # оптимизатор (можно "SGD" или "AdamW")
    lr0=0.01,               # начальная learning rate
    lrf=0.1,                # конечный LR = lr0 * lrf
    weight_decay=0.0005,     # L2-регуляризация
    augment=True,
    plots=True,
    cos_lr=True,             # косинусный спад LR
    warmup_epochs=3,         # число эпох тёплого старта
    warmup_momentum=0.8,     # начальный momentum в warmup (SGD) или beta1 для AdamW
    warmup_bias_lr=0.1,      # начальный LR для bias (будет спускаться к lr0)
    patience=6,              # ранняя остановка: число эпох без улучшения val mAP
    save_period=10,          # как часто (в эпохах) сохранять чекпойнты
    save=True,               # сохранять результаты и веса
    name="aitas-agro-detector",  # подпапка внутри project
    exist_ok=True,           # переписывать, если папка уже есть
    # workers=workers,               # число процессов для загрузки данных
    project="aitas-qcloudy/models",
    rect=True,
    # label_smoothing=0.01     # плавность меток для стабилизации обучения (deprecated-предупреждение можно игнорировать)
)