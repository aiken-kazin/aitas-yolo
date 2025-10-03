import json
from ultralytics import YOLO


MODEL_PATH = "models/best.pt" 

def predict(model_path, image_path):
    # Load model
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        imgsz=1280,          # change if you want
        conf=0.3,            # confidence threshold
        iou=0.5,             # NMS IoU
        device="cpu",        # GPU 0; use 'cpu' if no GPU
        save=True,           # save images with predictions
        show_labels=False,
        line_width=6,
        project="results",   # root folder
        name="sunflower",    # subfolder
        # show=True
    )

    output = []  # list to store JSON results for all images

    for i, r in enumerate(results):
        class_ids = r.boxes.cls.int().tolist()  # predicted class indices

        # initialize all classes with 0
        counts = {model.names[cls_id]: 0 for cls_id in model.names.keys()}

        # update counts
        for cls_id in class_ids:
            counts[model.names[cls_id]] += 1

        # store per-image result in JSON structure
        image_result = {
            "image": r.path,
            "counts": counts
        }
        output.append(image_result)

    # Print JSON formatted output
    print(json.dumps(output, indent=4, ensure_ascii=False))

    return output


if __name__ == "__main__":
    image_path = "test_image/DJI_0674.JPG"
    predict(MODEL_PATH, image_path)


