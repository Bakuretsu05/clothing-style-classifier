# 👕 Clothing Genre Classifier (CNN-based)

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

| Item                          | Description                                                               |
| :---------------------------- | :------------------------------------------------------------------------ |
| **Trained CNN model (`.h5`)** | Final TensorFlow/Keras model saved in HDF5 format.                        |
| **Gradio demo (`app.py`)**    | Interactive web interface for uploading an image and viewing predictions. |
| **Report**                    | Technical report describing dataset, model design, results, and analysis. |
| **Slides**                    | Presentation slides summarizing the workflow and findings.                |

---

## ⚙️ Model Configuration

| Parameter          | Value                                      | Note                                      |
| :----------------- | :----------------------------------------- | :---------------------------------------- |
| **Input size**     | 224×224 px, RGB                            | Standard ImageNet input dimension         |
| **Base CNN**       | MobileNetV3-Small (pretrained on ImageNet) | Lightweight and efficient for fine-tuning |
| **Framework**      | TensorFlow / Keras                         | Beginner-friendly and Colab-compatible    |
| **Output classes** | 7                                          | Clothing genre categories                 |
| **Optimizer**      | Adam                                       | Fast and stable training                  |
| **Loss function**  | Categorical Crossentropy                   | For multi-class classification            |
| **Metrics**        | Accuracy, F1-Score (per class)             | Balanced evaluation                       |

---

## 📈 Evaluation Metrics

-   **Accuracy** – measures overall correctness of predictions.
-   **F1-Score (per class)** – balances precision and recall for each clothing type.

---

## 🧑‍💻 Team Roles (suggested)

| Role               | Responsibility                                       |
| :----------------- | :--------------------------------------------------- |
| **Data Engineer**  | Collect and preprocess clothing images.              |
| **Model Engineer** | Implement and fine-tune the CNN.                     |
| **App Developer**  | Create and test the Gradio demo.                     |
| **Reporter**       | Write documentation and prepare presentation slides. |

---

## 🗓️ Timeline Summary

| Week | Focus                                 |
| :--- | :------------------------------------ |
| 1    | Project setup, scope, and environment |
| 2    | Dataset collection and cleaning       |
| 3–4  | Model development and training        |
| 5–6  | Fine-tuning and evaluation            |
| 7    | Demo integration (Gradio)             |
| 8    | Report and presentation preparation   |

---

## 🧰 Tech Stack

-   **Language:** Python 3.11
-   **Framework:** TensorFlow / Keras
-   **Base Model:** MobileNetV3-Small (ImageNet pretrained)
-   **Tools:** OpenCV, NumPy, Pandas, Matplotlib, Scikit-Learn, Gradio
-   **Dataset Sources:** DeepFashion 2, Clothing Dataset Full (Kaggle)

---

## 🧠 References

-   Hidayati, S.C. et al. (2012). _Clothing Genre Classification by Exploiting the Style Elements._
-   TensorFlow Transfer Learning Documentation
-   Gradio Library Documentation

---

> **Status:** Week 1 – Project Definition & Setup ✅  
> Next Step → Dataset Selection and Preprocessing (Week 2)
