This project focuses on detecting brain tumors using MRI scans with a deep learning model. By leveraging Convolutional Neural Networks (CNNs)
and K-Fold Cross-Validation, the model achieves high accuracy 96%.
Data Preprocessing:Resized MRI images to 120x120 pixels.,Applied normalization and data augmentation.
Model Architecture: 4-layer CNN with ReLU and Softmax activations. ,Max pooling layers for dimensionality reduction.

K-Fold Cross-Validation: Split dataset into 5 folds. Trained the model iteratively, testing each fold for robust evaluation.
Evaluation Metrics
Accuracy: Measures overall correctness.
Precision: Indicates true positive accuracy.
Recall: Captures the model's sensitivity.
F1 Score: Balances precision and recall.
Results and Findings:
Performance Metrics
Accuracy: 96%
Precision: 98%
Recall: 94%
F1 Score: 96%
Key finding:
The inclusion of data augmentation significantly reduced overfitting, and K-Fold Cross-Validation enhanced model reliability.
