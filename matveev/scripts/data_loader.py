import openml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_openml_data(
    dataset_id: int,
    test_size: float = 0.2,
    val_size: float = 0.5,
    seed: int = 0,
    transform_func=None
):
    dataset = openml.datasets.get_dataset(dataset_id=dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

    le = LabelEncoder()
    y = le.fit_transform(y)

    if transform_func is not None:
        X = transform_func(X, attribute_names, categorical_indicator)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
        shuffle=True
    )
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, 
        y_test, 
        test_size=val_size, 
        random_state=seed, 
        stratify=y_test,
        shuffle=True
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test, categorical_indicator, attribute_names