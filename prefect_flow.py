import subprocess
from pathlib import Path

from prefect import flow, task


@task(log_prints=True)
def preprocess_data() -> Path:
    """Run the preprocessing script and return the processed data path."""
    result = subprocess.run(
        ["python", "src/preprocess.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return Path("data/processed.csv")


@task(log_prints=True)
def train_model() -> Path:
    """Run the training script and return the model artifact path."""
    result = subprocess.run(
        ["python", "src/train.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return Path("models/model.pkl")


@task(log_prints=True)
def run_dvc_pipeline() -> None:
    """Run the DVC pipeline to ensure data and artifacts are current."""
    result = subprocess.run(
        ["dvc", "repro"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


@flow(name="mlops_orchestration")
def orchestrate_pipeline(use_dvc: bool = False):
    """Orchestrate preprocessing, optional DVC workflow, and model training."""
    processed_path = preprocess_data()

    if use_dvc:
        run_dvc_pipeline()

    model_path = train_model()
    return {
        "processed_data": str(processed_path),
        "model_path": str(model_path),
    }


if __name__ == "__main__":
    orchestrate_pipeline(use_dvc=False)
