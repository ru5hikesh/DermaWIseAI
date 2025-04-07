import os
import sys
import django
import json
from pathlib import Path

# Set up Django environment
sys.path.append(str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Import after Django setup
from predictor.prediction import predict_image
from predictor.bedrock_client import BedrockClient

def test_prediction(image_path):
    """Test the prediction pipeline with a local image file"""
    try:
        print(f"Testing prediction with image: {image_path}")
        
        # Open the image file
        with open(image_path, 'rb') as f:
            # Get prediction
            prediction_result = predict_image(f)
            print("\nPrediction Result:")
            print(f"Disease: {prediction_result['disease']}")
            print(f"Confidence: {prediction_result['confidence']:.2%}")
            
            # Get top predictions
            print("\nTop Predictions:")
            for pred in prediction_result.get('top_predictions', []):
                print(f"- {pred['disease']}: {pred['confidence']:.2%}")
            
            # Test Bedrock integration
            print("\nTesting Bedrock integration...")
            bedrock_client = BedrockClient()
            bedrock_result = bedrock_client.get_explanation(prediction_result)
            
            # Print explanation
            print("\nExplanation:")
            explanation = bedrock_result['explanation']
            for key, value in explanation.items():
                if key != 'raw_response':
                    print(f"\n{key.upper()}:")
                    print(value)
            
            # Combined result
            combined_result = {
                'disease': prediction_result['disease'],
                'confidence': prediction_result['confidence'],
                'top_predictions': prediction_result.get('top_predictions', []),
                'explanation': bedrock_result['explanation'],
            }
            
            # Save result to file for inspection
            output_file = 'test_prediction_result.json'
            with open(output_file, 'w') as f:
                json.dump(combined_result, f, indent=2)
            
            print(f"\nComplete result saved to {output_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python test_prediction.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_prediction(image_path)
