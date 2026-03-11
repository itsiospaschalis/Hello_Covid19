# Covid-19 scans
# Hello_Covid19 — COVID-19 Scan Classification Experiments

Portfolio-ready repository of **machine learning experiments** for **binary image classification** (*COVID* vs *Non‑COVID*) using transfer learning, plus a few supporting exploration scripts (PCA + GMM segmentation).

> **Note**: This repo contains **research/experiment scripts** (some written in a notebook-like style) and includes **hard-coded local paths** to datasets/checkpoints that are **not** part of the repository. The README below shows how to adapt the scripts to your environment.

---

## What I built

- **Binary classifier (COVID vs Non‑COVID)** using **transfer learning** with **VGG16** (TensorFlow/Keras).
- **Feature exploration** using **PCA** on grayscale image vectors (and notes toward PCA on deep features).
- **Image segmentation toy example** using a **Gaussian Mixture Model (GMM)** on pixel colors.
- **Checkpoint loading experiments** (e.g., loading pretrained/previously trained weights).

---

## Tech stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **scikit-learn**
- **OpenCV**
- **Matplotlib** (and optionally **Seaborn** for plots)

---

## Repository structure

- `transfer_learning_vgg.py`: VGG16 transfer learning workflow (data split into `train/valid/test`, training + evaluation, confusion matrix/ROC plotting).
- `load_greyscale_inception.py`: Loads InceptionV3 (without top) and attempts to restore weights from a checkpoint directory.
- `image_preprocessing_pca.py`: Grayscale preprocessing + PCA variance plot + additional PCA exploration snippets.
- `gmm.py`: Simple GMM-based image segmentation demo.
- `test.py`: TensorFlow v1-style checkpoint restore snippet.

---

## Dataset expectations

The scripts expect an image dataset organized like:

```text
<DATA_ROOT>/
  train/
    Covid/
    Non-Covid/
  valid/
    Covid/
    Non-Covid/
  test/
    Covid/
    Non-Covid/
```

Some scripts also start from a “flat” folder of images and then **move** a random sample into `train/valid/test`. If you don’t want files moved, copy the logic and replace `shutil.move(...)` with `shutil.copy(...)` (recommended).

---

## How to run (local)

### 1) Create an environment

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

There isn’t a pinned `requirements.txt` in this repo. A practical starting point is:

```bash
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib seaborn
```

If you run into version issues, match TensorFlow to your Python version (common on Windows).

### 3) Update paths inside scripts

Several files contain absolute paths like `C:\\Users\\itsios\\Desktop\\dissertation\\...`.

Search/replace these with your local locations:

- **Dataset root** (images)
- **Checkpoint directory / checkpoint file**

### 4) Run an experiment

Example:

```bash
python transfer_learning_vgg.py
```

---

## Results

The thesis behind this project evaluated **three different methods (without fine‑tuning)**. The main performance metrics on the test set are:

| **Metric**  | **Method 1** | **Method 2** | **Method 3** |
|------------|--------------|--------------|--------------|
| **Accuracy**  | 0.94 | 0.97 | **0.99** |
| **Precision** | 0.96 | 0.96 | **1.00** |
| **Recall**    | 0.92 | 0.98 | **0.98** |
| **F1‑score**  | 0.94 | 0.97 | **0.989** |
| **AUC**       | 0.94 | 0.97 | **0.99** |

**Method 3** achieved the best trade‑off across all metrics, with **99% accuracy**, **AUC 0.99**, and a **perfect precision of 1.00**, making it the strongest configuration among the three approaches tested in the thesis.

---

## Portfolio highlights

- **Transfer learning** approach to reduce training time and data requirements.
- **End-to-end experimentation**: preprocessing → training → evaluation metrics → visualization.
- **Model introspection ideas**: extracting intermediate layer outputs (e.g., VGG `fc2`) to study representations.

---

## Limitations / Notes

- Scripts are **experiment-oriented** and may include notebook-only directives (e.g., `%matplotlib inline`) and deprecated APIs.
- The code assumes external assets (datasets/checkpoints) not included in the repository.
- This project is **for educational/research purposes** and is **not a clinical tool**.

---

## License

If you plan to publish this on GitHub, add a license you’re comfortable with (e.g., MIT) and ensure the dataset (if any) is licensed for redistribution.
