import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_gbt(
        target_column,
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_split,
        train_size=0.7
):
    pass