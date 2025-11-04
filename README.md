# 👕 Wearly - Clothing Style Classifier

## 🧭 Project Overview

Wearly is a deep learning model that automatically classifies the **type of upper-body clothing** from a cropped human image.  
The system focuses on seven common clothing genres:

| Label | Description  |
| :---- | :----------- |
| 1     | T-shirt      |
| 2     | Polo         |
| 3     | Formal_Shirt |
| 4     | Tank_Top     |
| 5     | Sweater      |
| 6     | Hoodie       |
| 7     | Jacket       |

Using a pretrained **Convolutional Neural Network (CNN)** — specifically **MobileNetV3-Small** — the model will be fine-tuned on a labeled dataset of clothing images to recognize these categories accurately.  
The input to the model is a **cropped upper-body image**, resized to **224×224 pixels (RGB)**, consistent with ImageNet standards.

---

## 🚀 Complete Setup Guide

This guide will walk you through cloning the repository, setting up the environment, training the model, and running the demo from scratch.

---

### 📋 Prerequisites

Before you begin, ensure you have:

-   **Python 3.11+** (3.12 also works)
-   **Git** installed
-   **Conda** installed ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution))
-   **CUDA-capable GPU** (optional, but recommended for faster training)
-   **8GB+ RAM** (16GB recommended)
-   **~5GB free disk space** (for dataset and models)

---

### 📥 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/wearly.git

# Navigate to the project directory
cd wearly
```

---

### 🔧 Step 2: Set Up Conda Environment

```bash
# Create a new conda environment
conda create -n wearly python=3.11 -y

# Activate the environment
conda activate wearly

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OR for CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR for CPU-only (slower):
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install remaining dependencies
pip install -r requirements.txt
```

**Note:** If you encounter NumPy compatibility issues, install `numpy<2.0`:

```bash
pip install "numpy<2.0"
```

---

### 🔧 Step 3: Fix OpenMP Issue (Windows Only)

If you're on Windows, you may encounter an OpenMP error. Set this environment variable:

**Windows Command Prompt:**

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

**Windows PowerShell:**

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

**Linux/Mac:**

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

To make it permanent, add it to your shell profile (`.bashrc`, `.zshrc`, etc.) or set it as a system environment variable on Windows.

---

### 📊 Step 4: Prepare Your Dataset

The model expects data in CSV format with images in a separate folder.

#### Dataset Structure

Create the following directory structure:

```
wearly/
├── data/
│   └── raw/
│       ├── images.csv
│       └── images/
│           ├── image-uuid-1.jpg
│           ├── image-uuid-2.jpg
│           └── ...
```

#### CSV Format

Your `data/raw/images.csv` file should have the following columns:

| Column      | Description                          | Example                                |
| ----------- | ------------------------------------ | -------------------------------------- |
| `image`     | UUID or filename (without extension) | `4285fab0-751a-4b74-8e9b-43af05deee22` |
| `sender_id` | Sender identifier (not used)         | `124`                                  |
| `label`     | Clothing label                       | `T-Shirt`, `Polo`, `Shirt`, etc.       |
| `kids`      | Boolean flag (not used)              | `False`                                |

**Example CSV row:**

```csv
image,sender_id,label,kids
4285fab0-751a-4b74-8e9b-43af05deee22,124,T-Shirt,False
ea7b6656-3f84-4eb3-9099-23e623fc1018,148,Polo,False
```

#### Supported Labels

The model maps these labels to 7 classes:

| CSV Label                       | Mapped Class |
| ------------------------------- | ------------ |
| `T-Shirt`                       | T-shirt      |
| `Polo`                          | Polo         |
| `Shirt`, `Longsleeve`, `Blouse` | Formal_Shirt |
| `Undershirt`, `Top`             | Tank_Top     |
| `Body`                          | Sweater      |
| `Hoodie`                        | Hoodie       |
| `Outwear`, `Blazer`, `Jacket`   | Jacket       |

Labels like `Pants`, `Shoes`, `Dress`, `Not sure`, `Skip` will be filtered out.

#### Image Requirements

-   **Format:** JPG or PNG
-   **Location:** All images in `data/raw/images/` folder
-   **Naming:** Images should be named `{uuid}.jpg` (where `{uuid}` matches the `image` column in CSV)
-   **Quality:** Clear, cropped upper-body clothing images work best

---

### 🏋️ Step 5: Train the Model

Once your data is set up:

```bash
# Make sure your conda environment is activated
conda activate wearly

# Set OpenMP variable (if on Windows)
set KMP_DUPLICATE_LIB_OK=TRUE  # Windows CMD
# or
$env:KMP_DUPLICATE_LIB_OK="TRUE"  # PowerShell
# or
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

# Run training
python train.py
```

#### Training Configuration

The model is configured with:

-   **Epochs:** 50
-   **Batch Size:** 32
-   **Learning Rate:** 0.001
-   **Train/Val Split:** 80/20 (automatic)
-   **Model:** MobileNetV3-Small (pretrained on ImageNet)

#### Expected Training Time

-   **GPU (CUDA):** 15-30 minutes (depends on GPU model)
-   **CPU only:** 2-4 hours

#### Training Output

You'll see:

-   Dataset loading information
-   Training progress (loss, validation accuracy, F1-score)
-   Best model saved to `models/best_model.pth`
-   Training history plot saved to `training_history.png`

**Example output:**

```
Using device: cuda
Loaded 2385 images for train split
Loaded 597 images for val split
Model created: 1,234,567 parameters

Starting training...
Epoch [1/50], Loss: 1.8234, Val Acc: 0.4567, Val F1: 0.4321
  -> Saved best model with val acc: 0.4567
...
```

---

### 🎯 Step 6: Run the Demo

After training completes:

```bash
# Make sure conda environment is activated
conda activate wearly

# Set OpenMP variable (if on Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

# Launch the Gradio web interface
python app/demo.py
```

You should see:

```
Model loaded successfully from models/best_model.pth
Running on local URL:  http://0.0.0.0:7860

To create a public link, set `share=True` in `launch()`.
```

#### Access the Web Interface

Open your browser and navigate to:

-   **Local:** http://localhost:7860
-   **Network:** http://your-ip-address:7860

#### Using the Interface

1. **Upload an image** of upper-body clothing (drag & drop or click to browse)
2. The prediction will appear automatically
3. View:
    - **Predicted Class:** The most likely clothing type
    - **Confidence:** Percentage confidence
    - **Top 3 Predictions:** Alternative predictions with probabilities

---

### 🔍 Troubleshooting

#### Problem: "Model file not found"

**Solution:** Train the model first:

```bash
conda activate wearly
python train.py
```

#### Problem: "No training images found"

**Solution:** Check your data structure:

-   Ensure `data/raw/images.csv` exists
-   Ensure `data/raw/images/` folder exists with image files
-   Verify CSV format matches expected structure
-   Check that image filenames match the UUIDs in the CSV

#### Problem: OpenMP Error (Windows)

**Solution:** Set the environment variable:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

#### Problem: NumPy Compatibility Error

**Solution:** Install compatible NumPy version:

```bash
conda activate wearly
pip install "numpy<2.0"
```

#### Problem: CUDA Out of Memory

**Solution:** Reduce batch size in `train.py`:

```python
BATCH_SIZE = 16  # Change from 32 to 16 or lower
```

#### Problem: Port 7860 Already in Use

**Solution:** Change the port in `app/demo.py`:

```python
demo.launch(server_port=7861)  # Use different port
```

#### Problem: Slow Training on CPU

**Solution:**

-   Consider using a GPU (CUDA)
-   Reduce batch size
-   Reduce number of epochs for testing
-   Use Google Colab (free GPU available)

---

### 📝 Quick Start Checklist

-   [ ] Repository cloned
-   [ ] Conda environment created (`conda create -n wearly python=3.11 -y`)
-   [ ] Environment activated (`conda activate wearly`)
-   [ ] PyTorch installed (with or without CUDA)
-   [ ] Dependencies installed (`pip install -r requirements.txt`)
-   [ ] Dataset prepared (`data/raw/images.csv` and `data/raw/images/`)
-   [ ] OpenMP variable set (Windows)
-   [ ] Model trained (`python train.py`)
-   [ ] Demo running (`python app/demo.py`)
-   [ ] Web interface accessible at http://localhost:7860

---

## 🎯 Objectives

-   Build an image classifier capable of identifying upper-body clothing genres.
-   Apply **transfer learning** using a pretrained MobileNetV3-Small backbone.
-   Evaluate model performance using **Accuracy** and **per-class F1-Score**.
-   Develop an interactive **web demo** using **Gradio**.

---

## 📦 Deliverables

| Item                            | Description                                                               |
| :------------------------------ | :------------------------------------------------------------------------ |
| **Trained CNN model (`.pth`)**  | Final PyTorch model saved in checkpoint format.                           |
| **Gradio demo (`app/demo.py`)** | Interactive web interface for uploading an image and viewing predictions. |
| **Report**                      | Technical report describing dataset, model design, results, and analysis. |
| **Slides**                      | Presentation slides summarizing the workflow and findings.                |

---

## ⚙️ Model Configuration

| Parameter          | Value                                      | Note                                           |
| :----------------- | :----------------------------------------- | :--------------------------------------------- |
| **Input size**     | 224×224 px, RGB                            | Standard ImageNet input dimension              |
| **Base CNN**       | MobileNetV3-Small (pretrained on ImageNet) | Lightweight and efficient for fine-tuning      |
| **Framework**      | PyTorch                                    | Flexible and efficient deep learning framework |
| **Output classes** | 7                                          | Clothing genre categories                      |
| **Optimizer**      | Adam                                       | Fast and stable training                       |
| **Loss function**  | CrossEntropyLoss                           | For multi-class classification                 |
| **Metrics**        | Accuracy, F1-Score (per class)             | Balanced evaluation                            |

---

## 📈 Evaluation Metrics

-   **Accuracy** – measures overall correctness of predictions.
-   **F1-Score (per class)** – balances precision and recall for each clothing type.

---

## 🧰 Tech Stack

-   **Language:** Python 3.11
-   **Framework:** PyTorch
-   **Base Model:** MobileNetV3-Small (ImageNet pretrained)
-   **Tools:** OpenCV, NumPy, Pandas, Matplotlib, Scikit-Learn, Gradio
-   **Dataset Sources:** DeepFashion 2, Clothing Dataset Full (Kaggle)

---

## 🔗 Additional Resources

-   [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
-   [Gradio Documentation](https://gradio.app/docs/)
-   [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
-   [Conda Documentation](https://docs.conda.io/)

---

## 💡 Tips

1. **Better Results:** Use clear, cropped images of upper-body clothing
2. **Faster Training:** Use GPU if available
3. **Testing:** Reduce `NUM_EPOCHS` in `train.py` for quick testing
4. **Monitoring:** Watch the validation accuracy to see model improvement
5. **Model Saving:** Best model is automatically saved when validation accuracy improves
6. **Environment:** Always activate your conda environment before running scripts: `conda activate wearly`

---

## 🧠 References

-   Hidayati, S.C. et al. (2012). _Clothing Genre Classification by Exploiting the Style Elements._
-   PyTorch Transfer Learning Documentation
-   Gradio Library Documentation

---

## 📧 Support

If you encounter issues not covered here:

1. Check the error messages carefully
2. Verify all prerequisites are met
3. Ensure your dataset format is correct
4. Check GitHub Issues (if applicable)

---
