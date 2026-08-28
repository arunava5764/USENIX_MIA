# USENIX 2027 Cycle 1 Artifact

This repository contains the artifact for the paper:

**_Local Data, Global Attention: GPU Side-Channel Based Membership Inference Attack on Privacy-Preserving Vision Transformers_**  
*Submitted to the 36th USENIX Security Symposium (USENIX Security'27).*

This artifact provides a proof-of-concept implementation of our proposed GPU side-channel attack model. The included Jupyter Notebooks demonstrate how GPU attention-kernel performance counters can be used to perform Membership Inference attacks (MIA) against Vision Transformer (ViT) models.

This repository reproduces the experimental results reported in **Tables 4, 5, 6, 7, 8** of the paper, showing that our attack identifies ViT-Base and DeiT-Base models' training samples with competitive accuracy across three different NVIDIA GPU architectures. Also the **Proposed Mitigation** folder contains the results against our newly introduced temperature scaling-based mitigation approach.

---

## 📦 Repository Contents

- **`Table_X...`** — X denotes the table number in the paper. Each notebook in the folder:
  - Loads shadow model traces
  - Trains a Membership Inference classifier
  - Evaluates the attack on target model traces
  - Reproduces the corresponding results reported in the paper
- Utility code and processing logic (embedded in the notebook)

---

## 📂 Dataset Download

Both the shadow model traces and target model traces used in the paper can be downloaded from Zenodo:

👉 **Dataset link:** [https://zenodo.org/records/21994193](https://zenodo.org/records/21994193)

After downloading, extract the ZIP files inside the folder with the match initial of the downloaded data file.

---

### Software
- Python **3.10.12+**
- Jupyter Notebook / JupyterLab
- Standard Python ML/data libraries (NumPy, Scikit-learn, Pandas, Matplotlib, etc.)
- CUDA + CUPTI (only needed if collecting *new* traces; not required for running the provided ones)

---

## ▶️ How to Reproduce the Results

1. **Download & extract** the trace dataset from Zenodo.
2. Open the notebook:  Set the Dataset Path. Inside the notebook, locate the following placeholder in the 2nd code block:

```python
...........
for i in range(1,101):
    temp1=[]
    for j in range(10):
        path='./mem_csv_vit_base_gtsrb/report'+str(j)+'_'+str(i)+'.csv' //Shadow Model Traces
        data=pd.read_csv(path)
        temp1.append(data)
    vit_base_data.append(temp1)

for i in range(1,31):
    temp1=[]
    for j in range(10):
        path='./mem_csv_vit_base_gtsrb_test/report'+str(j)+'_'+str(i)+'.csv' //Target Model Traces
        data=pd.read_csv(path)
        temp1.append(data)
    vit_base_data_test.append(temp1)
.............
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

