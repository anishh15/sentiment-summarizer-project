import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visualizations")
CLASSIFICATION_REPORT_PATH = os.path.join(MODELS_DIR, "classification_report.csv")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")