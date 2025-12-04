# **Assignment 5 – Video Analysis**

**ECSE 415 – McGill University**

This repository contains the full solution for Assignment 5, including data preparation, YOLOv8 + DeepSORT tracking, MOTA evaluation, and pedestrian counting for the Kaggle competition.

---

# **1. Requirements & Libraries**

All code runs in Python 3.
Libraries used (as permitted by the assignment):

- `numpy`
- `opencv-python`
- `torch`
- `ultralytics` (YOLOv8)
- `deep-sort-realtime`
- `matplotlib` (optional)
- `pandas`
- `scipy` (for Hungarian algorithm)
- `torchvision` (for Faster R-CNN)

Make sure the following models/downloads are available:

- `yolov8x.pt`
- `fasterrcnn_resnet50_fpn_v2` weights (downloaded automatically by PyTorch)

GPU is recommended but not required.

---

# **2. How to Run the Notebook**

1. Place the dataset in the following directory structure:

```
Tracking/
 ├── Task1/
 │    ├── images/
 │    └── gt/gt.txt
 └── Task2/
      └── images/
```

2. Open the Jupyter notebook (`A5.ipynb`).

3. Run all cells **top to bottom**.
   The notebook will automatically produce:

   - `task1_input.mp4`
   - `task1.mp4`
   - `task1_tracks.txt`
   - `task2.mp4`
   - `task2_tracks.txt`
   - `task2_count.csv`

4. Submit `task2_count.csv` to Kaggle under your group name.

---

# **3. Summary of Implemented Tasks**

### **Task 1 – Data Preparation (10 pts)**

- Convert Task1 images to a 14 FPS video.
- Output: `task1_input.mp4`

### **Task 2 – YOLOv8 + DeepSORT Tracking (40 pts)**

- Run YOLOv8 for pedestrian detection.
- Use DeepSORT to maintain tracking IDs across frames.
- Save annotated tracking video and MOT-format results.
- Outputs:

  - `task1.mp4`
  - `task1_tracks.txt`

### **Task 3 – MOTA Evaluation (40 pts)**

- Load `gt.txt` (using first 6 columns only).
- Use IoU ≥ 0.5 for matching.
- Hungarian algorithm for optimal assignment.
- Count FP, FN, IDSW, GT.
- Compute MOTA.
- Achieved MOTA: **0.47**

### **Task 4 – Prediction + Kaggle (10 pts)**

- Use Faster R-CNN + DeepSORT on Task2 data.
- Count detected persons per frame.
- Save CSV for Kaggle submission.
- Achieved Kaggle score: **2.38**
- Outputs:

  - `task2.mp4`
  - `task2_count.csv`

---

# **4. Notes on Performance**

### **MOTA = 0.47 (Task 1)**

Lower due to:

- missed small/occluded pedestrians (FN),
- extra detections (FP),
- ID switches in crowded scenes (IDSW).

### **Kaggle RMSE = 2.38 (Task 2)**

Errors mainly caused by:

- duplicate detections in crowded frames,
- missed small pedestrians,
- non-optimized detector threshold.

---

# **5. Submission Checklist**

✔ Jupyter Notebook (.ipynb) with all code, outputs, and explanations
✔ `task1_input.mp4`
✔ `task1.mp4`
✔ `task2.mp4`
✔ `task2_count.csv`
✔ Kaggle submission under your group name
✔ Markdown explanations included
✔ Code runs without errors
