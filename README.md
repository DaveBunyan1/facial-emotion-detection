# Facial Emotion Detection

Deep learning project for facial emotion classification using CNNs, VGG16 transfer learning, and Optuna hyperparameter optimization.

## Overview

This project explores the use of deep learning to classify facial images into four emotion categories:

- Happy
- Sad
- Surprise
- Neutral

Six different model approaches were evaluated, including three custom CNN architectures and three transfer-learning approaches using **VGG16, ResNet, and EfficientNet**.

VGG16 was ultimately selected as the most promising architecture based on its performance and ability to achieve higher accuracy without excessive overfitting.

## Approach

The project included:

- Exploratory analysis of the image dataset
- Development and comparison of multiple CNN architectures
- Transfer learning using VGG16 pretrained on ImageNet
- Hyperparameter optimization using **Optuna** over 50 trials
- Fine-tuning of the selected VGG16 model
- Evaluation using accuracy and class-level F1 scores
- Analysis of model errors and potential overfitting
- Investigation of possible approaches for further improvement

The final VGG16 model used transfer learning up to the `block5_conv1` layer, followed by a fully connected layer and dropout.

## Results

The final model achieved approximately **81% accuracy** on the test set.

However, performance varied substantially between emotion classes. The model performed particularly well on the **surprise** and **happy** classes, while other classes were more difficult to classify.

The results also indicated potential overfitting and class imbalance, meaning that overall accuracy alone did not provide a complete picture of model performance.

## Conclusion

VGG16 was selected as the model to proceed with based on the experiments conducted.

Further investigation of incorrectly classified images, particularly images predicted as **neutral**, could provide insight into the model's errors. Adjusting classification thresholds and gradually increasing model complexity are potential approaches for improving performance.

Additional hyperparameter tuning and, most importantly, a larger and more representative dataset would also be valuable for improving generalization.

## Limitations

The original dataset used for this project is no longer available to me, so the model cannot currently be retrained from this repository.

The project was originally developed and executed in **Google Colab**. The Jupyter notebook is preserved with its original code, analysis, visualizations, and results.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Optuna
- Matplotlib
- Convolutional Neural Networks
- Transfer Learning
- VGG16

## Project Notebook

The complete analysis, experiments, code, visualizations, and results are available in the Jupyter notebook:

**[`facial_emotion_detection.ipynb`](./facial_emotion_detection.ipynb)**

---

_Developed as part of MIT Artificial Intelligence coursework._
