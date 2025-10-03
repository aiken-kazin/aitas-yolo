from ultralytics import YOLO



MODEL_PATH = "/aitas-qcloudy/models/best.pt" 

def predict(model_path, image_path):
    # Load a model
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        imgsz=1280,          # change if you want
        conf=0.3,            # confidence threshold
        iou=0.5,             # NMS IoU
        device=0,            # GPU 0; use 'cpu' if no GPU
        save=True,           # save images with predictions
        show_labels=False,
        line_width=6,
        project="results",   # your root folder
        name="sunflower",    # subfolder inside project
        # show=True          # display the image with predictions
    )

    for i, r in enumerate(results):
        class_ids = r.boxes.cls.int().tolist()  # predicted class indices

        # initialize all classes with 0
        counts = {cls_id: 0 for cls_id in model.names.keys()}

        # update counts
        for cls_id in class_ids:
            counts[cls_id] += 1

        print(f"\nImage {i+1}: {r.path}")
        for cls_id, cnt in counts.items():
            class_name = model.names[cls_id]
            print(f"  {class_name}: {cnt}")



if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    predict(MODEL_PATH, image_path)


