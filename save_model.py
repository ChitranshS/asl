import mlflow
import mlflow.keras
from keras.models import load_model

model = load_model('Model/smnistkaggle.h5')  # Your existing model path

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")  # Local tracking server
mlflow.set_experiment("ASL_Detection")

# Log model
with mlflow.start_run():
    mlflow.keras.log_model(
        model,
        "model",
        registered_model_name="asl_cnn",
        conda_env={
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.9',
                'tensorflow',
                'mlflow',
                'numpy',
                'pillow'
            ]
        }
    )
    print("Model logged successfully!")