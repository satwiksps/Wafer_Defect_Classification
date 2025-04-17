## Wafer Defect Classification: CNN vs Transformer vs Quantum ML

This project compares three state-of-the-art approaches—**Convolutional Neural Networks (CNNs)**, **Vision Transformers**, and **Quantum Machine Learning (QML)**—for the task of **wafer defect classification** using semiconductor wafer map images. It provides a benchmark-style analysis of accuracy, speed, and model efficiency on the MixedWM38 dataset.

---

## 🚀 Models Compared

### 1. 🧱 CNN (Baseline)
- Architecture: Custom 5-layer CNN
- Framework: PyTorch
- Strengths: Lightweight, fast training, well-established

### 2. 🔍 Vision Transformers
- Models Used: `MobileViTV2`, `LeViT`, `TinyViT`, `SwiftFormer`
- Framework: PyTorch
- Highlights: Token-based attention, better global context understanding

### 3. ⚛️ Quantum ML
- Tools: Qiskit / PennyLane
- Strategy: Hybrid quantum-classical classifier
- Limitation: Limited input size (quantum embedding), longer training time
- Goal: Explore early-stage viability for wafer classification

---

## 📊 Performance Metrics

| Model         | Accuracy | F1 Score | Params | Inference Time |
|---------------|----------|----------|--------|----------------|
| CNN           | XX%      | XX       | ~XXK   | XX ms/image    |
| LeViT         | XX%      | XX       | ~XXK   | XX ms/image    |
| SwiftFormer   | XX%      | XX       | ~XXK   | XX ms/image    |
| QML (Hybrid)  | XX%      | XX       | N/A    | XX ms/image    |

> Replace XX with actual results from your experiments.

---

## 🛠 Setup

```bash
git clone https://github.com/yourusername/Wafer_Defect_Classification.git
cd Wafer_Defect_Classification
pip install -r requirements.tx
```
# 🧪 Run Experiments
#### CNN
```bash
Copy
Edit
python models/cnn_model.py
```
#### Vision Transformers
```bash
Copy
Edit
python models/vit_model.py
```
#### Quantum ML
```bash
Copy
Edit
# Use Jupyter to run:
jupyter notebook models/qml_model.ipynb
```
# 📈 Dataset
MixedWM38: A challenging wafer defect classification dataset with 38 categories.

# 📜 License
MIT License. Feel free to use and adapt with credit.
