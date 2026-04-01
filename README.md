# Training Datasets

The trained models used in this project are **not publicly distributed**.  
Instead, you can access the original datasets and retrain the models yourself using Roboflow.

The datasets have been manually annotated & manually obtained from Youtube dashcam videos & car spot sites.
Please keep in mind that the dataset only contains Dutch license plates. Which differ font/color wise from other European plates.

---

## OCR Dataset (Character Detection)

Used for recognizing individual license plate characters:

https://app.roboflow.com/license-plate-detection-habec/croppedplatesocrv3/models

---

## License Plate Detection Dataset

Used to detect license plates within vehicle crops:

  https://app.roboflow.com/license-plate-detection-habec/license-plate-detection-ekjen/2

---

## Vehicle Detection Dataset

Used to detect vehicles in the scene before plate detection:
 
  https://app.roboflow.com/license-plate-detection-habec/vehicledetectiondashcam/models

---

## Notes on Training

These datasets follow a typical **ANPR (Automatic Number Plate Recognition)** pipeline for long distance detection:

1. Vehicle detection  
2. Plate detection  
3. Character-level OCR (Even though Yolo is not recommended for OCR, for some reason it was more accurate & less laggy than actual OCR models such as TrOCR and EasyOCR)  

You will need to:

- Export datasets from Roboflow (YOLOv8 format recommended, other versions might affect performance)
- Train using Ultralytics YOLO
- Replace the `.pt` weight paths in the script

---

## Why Models Are Not Included

The trained weights are excluded because redistribution violates Roboflow's platform terms


---

## How to Reproduce Models

1. Open the dataset links above  
2. Export dataset in YOLO format  
3. Train on a droplet or your own device using Ultralytics (or alternatively with Roboflow premium):

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=5
