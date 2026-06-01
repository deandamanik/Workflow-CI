import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def main():
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
    if "MLFLOW_EXPERIMENT_NAME" in os.environ:
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "tokopedia_preprocessing", "train_clean.csv")
    test_path = os.path.join(base_dir, "tokopedia_preprocessing", "test_clean.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df['text'] = train_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')

    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df['text'])
    X_test = tfidf.transform(test_df['text'])
    y_train = train_df['label']
    y_test = test_df['label']

    workspace_dir = os.environ.get("GITHUB_WORKSPACE", os.path.normpath(os.path.join(base_dir, "..")))
    local_model_path = os.path.join(workspace_dir, "model_output")

    with mlflow.start_run(run_name="CI_Automated_Retraining"):
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_param("C", 1.0)
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Simpan lokal di luar folder isolasi menggunakan path mutlak workspace utama
        os.makedirs(local_model_path, exist_ok=True)
        mlflow.sklearn.save_model(model, path=local_model_path)

        print(f"CI Retraining Berhasil! Path: {local_model_path} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()