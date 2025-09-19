# Image Classification using Texture Features and SVM

This project demonstrates a classic machine learning approach for image classification. It uses texture feature extraction methods—**Gray Level Co-occurrence Matrix (GLCM)** and **Local Binary Patterns (LBP)**—to classify land use images from the UC Merced dataset using a Support Vector Machine (SVM) classifier.

## Description

The script performs the following key tasks:

1.  **Loads an image dataset** organized into class-specific folders.
2.  For each image, it **extracts two types of texture features**:
      * **GLCM features**: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM.
      * **LBP features**: A histogram of uniform patterns.
3.  **Combines these features** into a single feature vector for each image.
4.  **Splits the data** into training and testing sets.
5.  **Scales the features** to standardize their range.
6.  **Optimizes the SVM classifier's hyperparameters** (`C` and `gamma`) using `GridSearchCV` for the best performance.
7.  **Trains the final SVM model** on the training data.
8.  **Evaluates the model's accuracy** on the unseen test data and prints the results.

## Key Features

  * **Feature Engineering**: Combines global (GLCM) and local (LBP) texture features to create a robust image representation.
  * **Machine Learning Model**: Utilizes a Support Vector Machine (SVM), a powerful and effective classifier.
  * **Hyperparameter Tuning**: Automatically finds the best model parameters using `GridSearchCV` to maximize performance.
  * **Dataset**: Designed for the popular UC Merced Land Use dataset but can be adapted for any categorized image dataset.

## Requirements

You will need Python 3 and the following libraries. You can install them using pip:

```bash
pip install numpy opencv-python scikit-learn scikit-image matplotlib seaborn
```

## Dataset Setup

1.  **Download the Dataset**: This script is configured for the **UC Merced Land Use Dataset**. You can download it from the [official source](https://www.google.com/search?q=http://vision.ucmerced.edu/datasets/landuse.html).

2.  **Directory Structure**: After unzipping, ensure your directory structure looks like this:

    ```
    .
    ├── your_script_name.py
    └── UCMerced_LandUse/
        └── Images/
            ├── agricultural/
            │   ├── agricultural00.tif
            │   └── ...
            ├── airplane/
            │   ├── airplane00.tif
            │   └── ...
            └── ... (other class folders)
    ```

3.  The script expects the image files to have a `.tif` extension, as seen in the original dataset.

##  How to Run

1.  Make sure all the required libraries are installed.
2.  Place the dataset in the correct directory structure as described above.
3.  If your dataset folder is named differently, update the `dataset_path` variable in the script:
    ```python
    # In the main execution block
    dataset_path = "path/to/your/Images"
    ```
4.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```

## Expected Output

The script will print its progress, including feature extraction for each class, the results of the hyperparameter tuning, and the final model accuracy. Based on the output you provided, the results are:

```
Feature extraction started ---
Processing class: agricultural (100 images)
Processing class: airplane (100 images)
...
Feature extraction completed.
Total samples: 1600, Features per sample: 16
Training set size: 1200
Testing set size: 400

Starting SVM hyperparameter tuning with GridSearchCV...
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Tuning complete.
Best parameters found:  {'C': 100, 'gamma': 'scale', 'kernel': 'rbf'}

Evaluating the best model on the test set...
Model Accuracy: 77.50%
```

This indicates that the best-performing SVM model achieved an accuracy of **77.50%** on the test set.
