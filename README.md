# 👕 Clothing Style Classifier (CNN-based)

## 🧭 Project Overview

This project aims to develop a deep learning model that automatically classifies the **type of upper-body clothing** from a cropped human image.  
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

## 🎯 Objectives

-   Build an image classifier capable of identifying upper-body clothing genres.
-   Apply **transfer learning** using a pretrained MobileNetV3-Small backbone.
-   Evaluate model performance using **Accuracy** and **per-class F1-Score**.
-   Develop an interactive **web demo** using **Gradio**.

---

## 📦 Deliverables

| Item                           | Description                                                               |
| :----------------------------- | :------------------------------------------------------------------------ |
| **Trained CNN model (`.pth`)** | Final PyTorch model saved in checkpoint format.                           |
| **Gradio demo (`app.py`)**     | Interactive web interface for uploading an image and viewing predictions. |
| **Report**                     | Technical report describing dataset, model design, results, and analysis. |
| **Slides**                     | Presentation slides summarizing the workflow and findings.                |

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

## 🧠 References

-   Hidayati, S.C. et al. (2012). _Clothing Genre Classification by Exploiting the Style Elements._
-   PyTorch Transfer Learning Documentation
-   Gradio Library Documentation

---
