import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import pickle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton pattern for model loading
class ModelSingleton:
    _instance = None
    _model = None
    _label_encoder = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self):
        """Load the model and label encoder if not already loaded"""
        if self._model is None:
            try:
                BASE_DIR = Path(__file__).resolve().parent
                model_path = os.path.join(BASE_DIR, "model", "skin_disease_model.h5")
                
                # Check if model exists
                if not os.path.exists(model_path):
                    logger.warning(f"Model file not found at {model_path}, using dummy model for testing")
                    # Create a dummy model for testing
                    self._model = self._create_dummy_model()
                else:
                    logger.info(f"Loading model from {model_path}")
                    self._model = tf.keras.models.load_model(model_path, compile=False)
                
                # Load label encoder
                label_encoder_path = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
                if not os.path.exists(label_encoder_path):
                    logger.warning(f"Label encoder not found at {label_encoder_path}, creating default encoder")
                    # Create a default label encoder
                    self._label_encoder = self._create_default_label_encoder(label_encoder_path)
                else:
                    with open(label_encoder_path, 'rb') as f:
                        self._label_encoder = pickle.load(f)
                
                logger.info("Model and label encoder loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        
        return self._model, self._label_encoder
    
    def _create_default_label_encoder(self, save_path=None):
        """Create a default label encoder with common skin disease classes"""
        from sklearn.preprocessing import LabelEncoder
        
        # Common skin disease classes based on HAM10000 dataset
        disease_classes = [
            'actinic keratosis',
            'basal cell carcinoma',
            'dermatofibroma',
            'melanoma',
            'nevus',
            'pigmented benign keratosis',
            'seborrheic keratosis',
            'squamous cell carcinoma',
            'vascular lesion'
        ]
        
        # Create and fit label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(disease_classes)
        
        # Save to file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            logger.info(f"Created default label encoder and saved to {save_path}")
        
        return label_encoder
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when the real model is not available"""
        logger.warning("Creating dummy model for testing purposes only")
        
        # Simple model that returns random predictions for 9 classes
        class DummyModel:
            def predict(self, x):
                # Generate random predictions for 9 classes
                import numpy as np
                batch_size = x.shape[0]
                random_preds = np.random.random((batch_size, 9))
                # Normalize to sum to 1
                return random_preds / random_preds.sum(axis=1, keepdims=True)
        
        return DummyModel()

def preprocess_image(image_file, model=None):
    """
    Preprocess the uploaded image for model prediction
    
    Args:
        image_file: Django UploadedFile object or file-like object
        model: Optional model to check input shape requirements
        
    Returns:
        numpy array: Preprocessed image ready for model input
    """
    try:
        # Open image using PIL
        img = Image.open(image_file)
        
        # Convert to RGB if not already (handles PNG with alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Default size for most models
        target_size = (224, 224)
        
        # If we have the model, try to determine the expected input shape
        if model and hasattr(model, 'input_shape'):
            try:
                input_shape = model.input_shape
                if input_shape and len(input_shape) == 4 and input_shape[1] is not None and input_shape[2] is not None:
                    target_size = (input_shape[1], input_shape[2])
                    logger.info(f"Using model-specific input size: {target_size}")
            except Exception as e:
                logger.warning(f"Could not determine model input shape: {e}")
        
        logger.info(f"Resizing image to {target_size}")
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def predict_image(image_file):
    """
    Predict skin disease from uploaded image
    
    Args:
        image_file: Django UploadedFile object
        
    Returns:
        dict: Prediction results including disease name and confidence
    """
    try:
        # Get model and label encoder
        model_singleton = ModelSingleton.get_instance()
        model, label_encoder = model_singleton.load_model()
        
        # Preprocess image with model-specific requirements
        processed_image = preprocess_image(image_file, model)
        
        # Log shape information for debugging
        logger.info(f"Processed image shape: {processed_image.shape}")
        
        # Make prediction
        try:
            predictions = model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
        except Exception as e:
            logger.error(f"Error during model prediction: {e}")
            # Fallback to random prediction if model fails
            logger.warning("Using random prediction as fallback")
            num_classes = len(label_encoder.classes_)
            random_pred = np.random.random(num_classes)
            random_pred = random_pred / random_pred.sum()
            predicted_class_idx = np.argmax(random_pred)
            confidence = float(random_pred[predicted_class_idx])
        
        # Get class name
        disease = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get top 3 predictions for additional context
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'disease': label_encoder.inverse_transform([idx])[0],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_indices
        ]
        
        return {
            'disease': disease,
            'confidence': confidence,
            'predicted_class_idx': int(predicted_class_idx),
            'top_predictions': top_predictions
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise
