import os
import logging
from django.shortcuts import render
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .prediction import predict_image
from .bedrock_client import BedrockClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionView(APIView):
    """API endpoint for skin disease prediction"""
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        """Handle POST requests with image uploads"""
        try:
            # Get the uploaded image file
            image_file = request.FILES.get('image')
            
            # Validate the image file
            if not image_file:
                return Response(
                    {'error': 'No image provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check file type
            allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
            if image_file.content_type not in allowed_types:
                return Response(
                    {'error': f'Unsupported file type: {image_file.content_type}. Please upload JPEG or PNG images only.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Check file size (limit to 10MB)
            if image_file.size > 10 * 1024 * 1024:  # 10MB
                return Response(
                    {'error': 'Image size too large. Please upload an image smaller than 10MB.'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            logger.info(f"Processing image upload: {image_file.name}")
            
            # Get prediction from model
            prediction_result = predict_image(image_file)
            
            # Get explanation from Bedrock
            bedrock_client = BedrockClient()
            bedrock_result = bedrock_client.get_explanation(prediction_result)
            
            # Combine results - only include the top disease as requested
            # Pass through the raw Bedrock response without modification
            combined_result = {
                'disease': prediction_result['disease'],
                'confidence': prediction_result['confidence'],
                'explanation': bedrock_result['explanation']
            }
            
            logger.info(f"Prediction complete: {prediction_result['disease']} with {prediction_result['confidence']:.2f} confidence")
            
            return Response(combined_result, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}")
            return Response(
                {'error': f'An error occurred during prediction: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
