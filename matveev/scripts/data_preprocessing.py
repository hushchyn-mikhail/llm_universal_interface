from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


def create_preprocessing_pipeline(attribute_names, categorical_indicator):
    categorical_features = [name for name, is_cat in zip(attribute_names, categorical_indicator) if is_cat]
    numerical_features = [name for name, is_cat in zip(attribute_names, categorical_indicator) if not is_cat]
    column_transformer = ColumnTransformer([
         ("ohe", OneHotEncoder(handle_unknown="ignore"), categorical_features),
         ("scaling", StandardScaler(), numerical_features)
    ])
    return column_transformer, categorical_features, numerical_features

def preprocess_for_nn(X_train, X_valid, X_test, categorical_features, numerical_features):
    for col in categorical_features:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_valid[col] = le.transform(X_valid[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    scaler = StandardScaler()
    if numerical_features:
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_valid[numerical_features] = scaler.transform(X_valid[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    
    return X_train, X_valid, X_test

def transform_to_sequence(X, attribute_names, categorical_indicator):
    categorical_cols = [name for name, is_cat in zip(attribute_names, categorical_indicator) if is_cat]
    numerical_cols = [name for name, is_cat in zip(attribute_names, categorical_indicator) if not is_cat]

    scaler = StandardScaler()
    if numerical_cols:
        X_numerical = scaler.fit_transform(X[numerical_cols])
    else:
        X_numerical = np.empty((X.shape[0], 0))
    
    X_categorical = []
    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(X[col])
        X_categorical.append(encoded)
    
    X_combined = np.column_stack([X_numerical] + X_categorical)

    X_seqs = [row.reshape(len(row), 1) for row in X_combined]
    return np.array(X_seqs)