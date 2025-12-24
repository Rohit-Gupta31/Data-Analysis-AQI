
# Histopathological Cancer Detection Using Deep Learning

## Project Overview

Histopathological Cancer Detection is a deep learning-based approach to classifying cancerous and non-cancerous tissue from histopathological images. By leveraging state-of-the-art techniques in image classification, this project aims to improve diagnostic accuracy and assist medical professionals in detecting cancer more efficiently.

The model employs **ResNet50** as its base architecture, fine-tuned on histopathological images for binary classification.

---

## Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)
* [Dataset](https://www.google.com/search?q=%23dataset)
* [Installation](https://www.google.com/search?q=%23installation)
* [Model Architecture](https://www.google.com/search?q=%23model-architecture)
* [Training and Evaluation](https://www.google.com/search?q=%23training-and-evaluation)
* [Results](https://www.google.com/search?q=%23results)
* [Contributions](https://www.google.com/search?q=%23contributions)
* [Future Work](https://www.google.com/search?q=%23future-work)
* [License](https://www.google.com/search?q=%23license)

---

## Dataset

* **Source**: The dataset used for this project comes from the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) competition on Kaggle.
* **Data**: The dataset consists of labeled **96x96 pixel** images in `.tif` format.
* **Classes**: Binary classification (**1**: Cancerous, **0**: Non-cancerous).
* **Data Augmentation**: To improve model robustness, several augmentation techniques such as rotation, zoom, shear, and horizontal flips are applied.

### Label Distribution

The label distribution is imbalanced, with more non-cancerous images than cancerous. To account for this, **stratified sampling** was used during the train-validation split.

---

## Installation

To get started, ensure you have **Python 3.x** installed.

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/histopathological-cancer-detection.git
cd histopathological-cancer-detection

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## Model Architecture

The model is based on **ResNet50**, a powerful residual convolutional neural network known for its ability to train very deep architectures using skip connections.

The following layers were added to fine-tune the model:

* **GlobalAveragePooling2D**: Reduces the spatial dimensions of the feature maps.
* **Dense Layer**: A fully connected layer with ReLU activation for high-level feature extraction.
* **Dropout Layer**: Applied to prevent overfitting.
* **BatchNormalization**: Introduced for improved convergence and model generalization.
* **Sigmoid Output Layer**: Produces binary classification predictions.

### Fine-tuning

After freezing the base ResNet50 model, additional layers were trained to adapt the model to the specific problem. Dropout and batch normalization were added to improve generalization.

---

## Training and Evaluation

The model was trained using **Binary Crossentropy** loss and the **Adam optimizer**.

### Training Process

1. **Data Preprocessing**: Pixel values were normalized to a  range.
2. **Train-Validation Split**: 80% training and 20% validation using stratified sampling.
3. **Model Training**: Trained for 20 epochs with an **Early Stopping** mechanism (patience of 5 epochs).
4. **Callbacks**: Used `ReduceLROnPlateau` to decrease the learning rate when validation accuracy stalled.

### Performance Metrics

* **Accuracy**: 80%+
* **Precision**: Focused on minimizing false positives.
* **Recall**: Monitored to ensure the model captures as many positive cases as possible.

---

## Results

The model achieved the following results on the validation set:

| Metric | Score |
| --- | --- |
| **Validation Accuracy** | 0.8040 |
| **Precision** | 0.8125 |
| **Recall** | 0.7921 |

---

## Future Work

* **Model Optimization**: Hyperparameter tuning for dropout rates, batch sizes, and learning rates.
* **Explainability**: Integrate **Grad-CAM** to visualize which regions of the image influence the model's decision.
* **Segmentation**: Move beyond classification to segment specific regions of interest.
* **Deployment**: Create a cloud-based API for real-time clinical inference.
* **Larger Input Sizes**: Experiment with  resolution to capture finer details.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](https://github.com/Swapnil-Verma24/Histopathalogic-Cancer-Detection/blob/main/LICENSE) file for more information.

---