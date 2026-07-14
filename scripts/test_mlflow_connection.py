import mlflow


DATABRICKS_PROFILE = "dbc-b0ed69bf-78a4"
EXPERIMENT_PATH = "/Shared/Reco-Nova-Experiments"


def main() -> None:
    mlflow.set_tracking_uri(
        f"databricks://{DATABRICKS_PROFILE}"
    )

    mlflow.set_experiment(EXPERIMENT_PATH)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", EXPERIMENT_PATH)

    with mlflow.start_run(
        run_name="vscode-connection-test"
    ) as run:
        mlflow.log_param("project", "Reco-Nova")
        mlflow.log_param("execution_environment", "VS Code")
        mlflow.log_metric("connection_success", 1.0)

        mlflow.set_tag("developer", "Peter")
        mlflow.set_tag("purpose", "connection-test")

        print("MLflow connection successful.")
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()