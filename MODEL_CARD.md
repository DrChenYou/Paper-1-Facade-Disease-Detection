# Model card: facade-defect detector

## Model description

This repository provides a YOLOv5s baseline protocol for localising and classifying two visible facade-defect categories: cracks and exterior-wall damage. It is a research companion implementation for the associated *Scientific Reports* article.

## Intended use

- research on computer-vision-assisted building inspection;
- preliminary screening of facade imagery;
- comparison of augmentation, detection, and edge-deployment strategies; and
- teaching reproducible object-detection workflows.

The detector is not a substitute for a qualified structural or facade inspection. A detected bounding box does not establish defect severity, cause, hidden deterioration, or structural safety.

## Inputs and outputs

- **Input:** image, video, webcam stream, or compatible folder accepted by YOLOv5.
- **Output:** bounding boxes, class labels, and confidence scores.
- **Classes:** `cracks`, `exterior_wall_damage`.

## Published evaluation

The associated paper reports precision 84.42%, recall 77.83%, F1 0.81, mAP@0.5 82.56%, and 55 FPS. These are paper-reported values. They are not benchmark claims for weights distributed in this repository because no weights are included.

## Known limitations

- small and geographically concentrated dataset;
- possible domain shift across architectural styles, facade materials, cameras, climates, and countries;
- sensitivity to low light, adverse weather, viewing distance, occlusion, shadows, cables, and vegetation;
- potential confusion between damage and similarly coloured background regions;
- two broad categories do not encode severity or repair urgency; and
- FPS is hardware- and implementation-dependent and should not be compared without a controlled benchmark.

## Evaluation recommendations

Report class-wise AP, precision-recall curves, confusion matrices, calibration, inference latency, and failure cases. Use building- or scene-grouped splits where possible. Include an external-city test set before claiming cross-city generalisation.

## Human oversight

All flagged regions should be reviewed by trained personnel. Decisions about access restrictions, repair, evacuation, or structural safety require appropriate professional assessment and evidence beyond this model.

