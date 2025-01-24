import numpy as np
from PIL import Image
import bentoml
from bentoml.io import Image as BentoImage, JSON

# Initialize service
asl_service = bentoml.Service("asl_classifier", runners=[])

# Define class labels
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

@asl_service.api(input=BentoImage(), output=JSON())
async def classify(input_image: Image.Image) -> dict:
    # Preprocess image (must match your training pipeline)
    img = input_image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))      # Resize to 28x28
    
    # Convert to numpy array
    img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Get prediction
    runner = bentoml.mlflow.get("asl_cnn:latest").to_runner()
    prediction = await runner.predict.async_run(img_array)
    
    # Process results
    return {
        "prediction": LETTERS[np.argmax(prediction)],
        "confidence": float(np.max(prediction)),
        "all_scores": {LETTERS[i]: float(prediction[0][i]) for i in range(len(LETTERS))}
    }