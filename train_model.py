import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
sys.stdout.reconfigure(encoding='utf-8')

def process_data(file_path, target_column="HeartDisease"):
    df = pd.read_csv(file_path)
    df.drop(columns=["Fid"], errors='ignore', inplace=True)
    
    # Handle missing values
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df.drop_duplicates(inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, label_encoders, scaler

def plot_classification_metrics(y_test, y_pred, model_name):
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose().drop(['accuracy'], errors='ignore')

    metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5))
    plt.title(f"Classification Metrics for {model_name}")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_name}_classification_metrics.png")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()


def plot_feature_importance(importances, model_name):
    importances.nlargest(10).plot(kind='barh', color='teal')
    plt.title(f"Top Features in {model_name}")
    plt.xlabel("Importance Score")
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{model_name}_feature_importance.png")
    plt.close()


def train_model(dataset_path, model_name, target_column="HeartDisease"):
    df, encoders, scaler = process_data(dataset_path, target_column)
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ {model_name} Accuracy: {acc:.2f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visuals
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_classification_metrics(y_test, y_pred, model_name)

    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    plot_feature_importance(feature_importance, model_name)

    # Save everything
    joblib.dump(model, f"{model_name}.pkl")
    joblib.dump(encoders, f"{model_name}_encoders.pkl")
    joblib.dump(scaler, f"{model_name}_scaler.pkl")
    joblib.dump(feature_importance, f"{model_name}_features.pkl")
    print(f"💾 {model_name} and related files saved!\n")

# Train models
train_model("Datasets.csv", "model_1")
train_model("heart_disease_self_measurable_pred.csv", "model_2")

print("🏁 All models trained and visualized.")
