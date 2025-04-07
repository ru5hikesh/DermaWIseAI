import boto3
import json
import os
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockClient:
    """Client for interacting with AWS Bedrock Agent"""
    
    def __init__(self):
        """Initialize the Bedrock client with credentials from environment variables"""
        try:
            # Initialize Bedrock client
            self.bedrock_agent = boto3.client(
                service_name='bedrock-agent-runtime',
                region_name=os.environ.get('AWS_REGION', 'ap-south-1'),
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            
            # Store agent ID
            self.agent_id = os.environ.get('AWS_BEDROCK_AGENT_ID', 'XLSPPD5OHN')
            self.agent_alias_id = os.environ.get('AWS_BEDROCK_AGENT_ALIAS_ID', 'LATEST')
            
            logger.info(f"Initialized Bedrock client with agent ID: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {e}")
            # Don't raise the exception to allow fallback behavior
    
    def get_explanation(self, prediction_result):
        """
        Get explanation from AWS Bedrock Agent
        
        Args:
            prediction_result: Dict containing prediction information
            
        Returns:
            dict: Explanation and additional information
        """
        try:
            # Create a detailed prompt for the Bedrock agent
            disease = prediction_result['disease']
            confidence = prediction_result['confidence']
            
            input_text = (
                f"Provide a detailed explanation about the skin disease '{disease}' "
                f"(detected with {confidence:.1%} confidence). "
                f"Include the following sections: "
                f"1. Brief description of the condition "
                f"2. Common symptoms "
                f"3. Causes and risk factors "
                f"4. Treatment options "
                f"5. When to see a doctor "
                f"Format the response as JSON with these section headings as keys."
            )
            
            logger.info(f"Sending request to Bedrock Agent for disease: {disease}")
            
            # Call Bedrock Agent
            response = self.bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                inputText=input_text
            )
            
            # Parse response
            completion = response.get('completion', '')
            
            # Try to extract JSON from the response if possible
            try:
                # Look for JSON-like content in the response
                if '{' in completion and '}' in completion:
                    start_idx = completion.find('{')
                    end_idx = completion.rfind('}') + 1
                    json_str = completion[start_idx:end_idx]
                    explanation_data = json.loads(json_str)
                else:
                    # If no JSON found, create a structured response
                    explanation_data = {
                        "description": completion,
                        "symptoms": "",
                        "causes": "",
                        "treatment": "",
                        "when_to_see_doctor": ""
                    }
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw text
                explanation_data = {
                    "description": completion,
                    "symptoms": "",
                    "causes": "",
                    "treatment": "",
                    "when_to_see_doctor": ""
                }
            
            # Add metadata
            explanation_data['source'] = 'AWS Bedrock Agent'
            explanation_data['disclaimer'] = (
                'This information is provided for educational purposes only and should not '
                'be considered medical advice. Please consult with a healthcare professional '
                'for proper diagnosis and treatment.'
            )
            
            return {
                'explanation': explanation_data,
                'raw_response': completion
            }
            
        except ClientError as e:
            logger.error(f"AWS Bedrock client error: {e}")
            return self._get_fallback_explanation(prediction_result)
            
        except Exception as e:
            logger.error(f"Error getting explanation from Bedrock: {e}")
            return self._get_fallback_explanation(prediction_result)
    
    def _get_fallback_explanation(self, prediction_result):
        """Provide a fallback explanation when Bedrock is unavailable"""
        disease = prediction_result['disease']
        
        return {
            'explanation': {
                'description': f"Information about {disease}",
                'symptoms': "Common symptoms information not available.",
                'causes': "Causes information not available.",
                'treatment': "Treatment information not available.",
                'when_to_see_doctor': "Please consult with a healthcare professional for proper diagnosis and treatment.",
                'source': 'Fallback system (Bedrock unavailable)',
                'disclaimer': (
                    'This information is provided for educational purposes only and should not '
                    'be considered medical advice. Please consult with a healthcare professional '
                    'for proper diagnosis and treatment.'
                )
            },
            'raw_response': ""
        }
