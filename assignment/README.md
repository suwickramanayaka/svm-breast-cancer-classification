# Machine Learning Assignment 01

Name: Sithum Wickramanayaka  
Index number: 269214N  
Batch: Batch 19

## Notebook

- `269214N_SVM_Assignment.ipynb` — complete executed notebook with explanations, tables and embedded plots.

## Reproduce the results

Open the notebook in Google Colab or Jupyter and run all cells in order. For a local Python environment, install the packages in requirements.txt. The dataset comes with scikit-learn; no external data file or private path is required. The notebook creates a results folder for exported figures and metrics.

The executed notebook used Python 3.12 and scikit-learn 1.9.1. An 80–20 stratified split and five-fold training cross-validation both use random_state=42. Scaling occurs inside the model pipeline. Malignant is label 1.

Linear SVM performs best on the observed test metrics. RBF is selected independently by training cross-validation ROC-AUC. Both confusion matrices are included in the notebook. The notebook explains this distinction.

## Supplementary visualization

The notebook also includes side-by-side Linear and RBF decision boundaries using mean radius and mean texture. These separate demonstration models use only training data, with C=1 and RBF gamma='scale'. They illustrate kernel behavior and do not replace the evaluated 30-feature models. The figure is exported to results/decision_boundaries.png.
