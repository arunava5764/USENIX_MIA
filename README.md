# VIT_MIA Artifact

This repository contains the artifact for the paper:

**_Local Data, Global Attention: GPU Side-Channel Based Membership Inference Attack on Privacy-Preserving Vision Transformers_**  
*Submitted to the 36th USENIX Security Symposium (USENIX Security'27).*

This artifact provides a proof-of-concept implementation of our proposed GPU side-channel attack model. The included Jupyter Notebook demonstrates how GPU attention-kernel performance counters can be used to perform Membership Inference (MI) attacks against Vision Transformer (ViT) models.

The notebook reproduces the experimental results reported in **Table 4** of the paper, showing that our attack identifies Google ViT-Base model training samples with **100% accuracy** on NVIDIA Ampere architecture GPUs when fine-tuned on **GTSRB**.

---

## 📦 Repository Contents

- **`ViT_membership_inference_gtsrb_POC.ipynb`** — Main proof-of-concept notebook  
  - Loads shadow model traces  
  - Trains Membership Inference classifier  
  - Evaluates attack on target model traces  
  - Reproduces Table 4 results
- Utility code and processing logic (embedded in the notebook)

---

## 📂 Dataset Download

Both the shadow model traces and target model traces used in the paper can be downloaded from Zenodo:

👉 **Dataset link:** https://zenodo.org/records/17661292

After downloading:

1. Extract the ZIP file.
2. Place the extracted directory somewhere accessible.
3. Note the full path to this directory, it will be used inside the notebook.

---

## 🛠️ System Requirements

### Hardware
- NVIDIA **Ampere** GPU (e.g., A100, RTX 3080/3090/3070) and post Ampere Generation GPUs
- CUPTI compatible NVIDIA driver

### Software
- Python **3.10+**
- Jupyter Notebook / JupyterLab
- Standard Python ML/data libraries (NumPy, Scikit-learn, Pandas, Matplotlib, etc.)
- CUDA + CUPTI (only needed if collecting *new* traces; not required for running the provided ones)

---

## ▶️ How to Reproduce the Results

1. **Download & extract** the trace dataset from Zenodo.
2. Open the notebook:  Set the Dataset Path. Inside the notebook, locate the following placeholder in the 2nd code block:

```python
"path to the directory"
```
3. Replace it with the actual path where you extracted the dataset.
4. Run the Notebook: From the Jupyter interface, select: Kernel → Restart & Run All. The notebook will automatically
     1. load the shadow model traces.
     2. Train the Membership Inference (MI) classifier.
     3. Evaluate the classifier on the target model traces.
     4. Output accuracy metrics, including the 100% classification accuracy reported in Table 4 (first row).
---

## 📑 Expected Output

After running all cells successfully, the notebook will generate:

1. Plots showing training and validation curves for the classifier
2. A confusion matrix and related accuracy metrics
3. Final Membership Inference accuracy matching the results reported in the paper

## 🎯 Supported Claims (per the Paper)

This artifact supports and validates the following claims:
1. GPU attention-kernel performance counters leak sufficient information to enable Membership Inference attacks.
2. Leakage characteristics differ between training and non-training samples.
3. On NVIDIA Ampere GPUs, this leakage enables near-perfect MI attacks against ViT-Base models.

