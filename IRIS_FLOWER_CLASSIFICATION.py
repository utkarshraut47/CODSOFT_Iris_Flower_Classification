import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing_columns)}")
    return df


def encode_species(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    encoder = LabelEncoder()
    df = df.copy()
    df["species_encoded"] = encoder.fit_transform(df["species"])
    return df, encoder


def split_dataset(df: pd.DataFrame):
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[features]
    y = df["species_encoded"]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def evaluate_model(model: SVC, X_test: pd.DataFrame, y_test: pd.Series, encoder: LabelEncoder) -> pd.DataFrame:
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
    print("\nConfusion Matrix:")
    print(cm_df)

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    results_df["Actual_Species"] = encoder.inverse_transform(results_df["Actual"])
    results_df["Predicted_Species"] = encoder.inverse_transform(results_df["Predicted"])
    return results_df


def plot_results(results_df: pd.DataFrame) -> None:
    actual_counts = (
        results_df["Actual_Species"].value_counts().rename_axis("Species").reset_index(name="Count")
    )
    actual_counts["Type"] = "Actual"

    predicted_counts = (
        results_df["Predicted_Species"].value_counts().rename_axis("Species").reset_index(name="Count")
    )
    predicted_counts["Type"] = "Predicted"

    plot_df = pd.concat([actual_counts, predicted_counts], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Species", y="Count", hue="Type", palette="viridis")
    plt.title("Distribution of Actual vs Predicted Species")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def main() -> None:
    dataset_path = Path(__file__).with_name("IRIS.csv")
    iris_df = load_dataset(dataset_path)

    print("First 5 rows of the dataset:")
    print(iris_df.head().to_string(index=False))
    print("\nDataset info:")
    iris_df.info()
    print("\nDescriptive statistics:")
    print(iris_df.describe(include="all"))

    iris_df, encoder = encode_species(iris_df)
    print("\nSpecies encoding mapping:")
    for index, species_name in enumerate(encoder.classes_):
        print(f"{species_name}: {index}")

    X_train, X_test, y_train, y_test = split_dataset(iris_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    svm_model = SVC(kernel="linear", random_state=42)
    svm_model.fit(X_train, y_train)
    print("\nSVC model trained successfully!")

    results_df = evaluate_model(svm_model, X_test, y_test, encoder)
    plot_results(results_df)


if __name__ == "__main__":
    main()
