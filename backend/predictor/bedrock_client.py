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
            # Get credentials from environment
            aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
            aws_session_token = os.environ.get('AWS_SESSION_TOKEN')
            aws_region = os.environ.get('AWS_REGION', 'ap-south-1')
            
            # Log credential availability (not the actual values)
            logger.info(f"AWS Region: {aws_region}")
            logger.info(f"AWS Access Key available: {bool(aws_access_key)}")
            logger.info(f"AWS Secret Key available: {bool(aws_secret_key)}")
            logger.info(f"AWS Session Token available: {bool(aws_session_token)}")
            
            # Initialize Bedrock client
            kwargs = {
                'service_name': 'bedrock-agent-runtime',
                'region_name': aws_region,
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key
            }
            
            # Add session token if available
            if aws_session_token:
                kwargs['aws_session_token'] = aws_session_token
                
            self.bedrock_agent = boto3.client(**kwargs)
            
            # Store agent ID and alias
            self.agent_id = os.environ.get('AWS_BEDROCK_AGENT_ID', 'XLSPPD5OHN')
            self.agent_alias_id = os.environ.get('AWS_BEDROCK_AGENT_ALIAS_ID', '8DF0G4GWOA')
            
            logger.info(f"Initialized Bedrock client with agent ID: {self.agent_id} and alias ID: {self.agent_alias_id}")
            
            # Initialize standard Bedrock client (not agent-specific)
            kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': aws_region,
                'aws_access_key_id': aws_access_key,
                'aws_secret_access_key': aws_secret_key
            }
            
            # Add session token if available
            if aws_session_token:
                kwargs['aws_session_token'] = aws_session_token
                
            self.bedrock = boto3.client(**kwargs)
            
            # Test connection with a simpler API call
            try:
                # List foundation models as a simpler test
                response = self.bedrock.list_foundation_models()
                logger.info(f"Successfully connected to AWS Bedrock. Found {len(response.get('modelSummaries', []))} foundation models.")
            except Exception as e:
                logger.error(f"AWS Bedrock connection test failed: {e}")
                # Don't raise the exception to allow fallback behavior
            
        except Exception as e:
            logger.error(f"Error initializing Bedrock client: {e}")
            # Don't raise the exception to allow fallback behavior
    
    def get_explanation(self, prediction_result):
        """
        Get explanation from AWS Bedrock
        
        Args:
            prediction_result: Dict containing prediction information
            
        Returns:
            dict: Explanation and additional information
        """
        try:
            # Create a detailed prompt about the skin disease
            disease = prediction_result['disease']
            confidence = prediction_result['confidence']
            
            # Try direct model invocation instead of agent
            prompt = (
                f"You are a dermatology expert. Provide a detailed explanation about the skin disease '{disease}' "
                f"(detected with {confidence:.1%} confidence). Include information about symptoms, causes, treatment, and when to see a doctor."
            )
            
            logger.info(f"Sending request to Bedrock for disease: {disease}")
            
            # Use Claude model directly instead of agent
            try:
                # Try using Claude model directly
                response = self.bedrock.invoke_model(
                    modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "temperature": 0.2,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ]
                    }),
                    contentType='application/json',
                    accept='application/json'
                )
                
                # Parse response
                response_body = json.loads(response.get('body').read())
                completion = response_body.get('content', [{}])[0].get('text', '')
                logger.info("Successfully received response from Claude model")
                
            except Exception as e:
                logger.error(f"Claude model invocation failed: {e}")
                # Try agent as fallback
                try:
                    # Fall back to agent invocation
                    logger.info("Falling back to agent invocation")
                    response = self.bedrock_agent.invoke_agent(
                        agentId=self.agent_id,
                        agentAliasId=self.agent_alias_id,
                        inputText=prompt
                    )
                    completion = response.get('completion', '')
                except Exception as agent_error:
                    logger.error(f"Agent invocation also failed: {agent_error}")
                    return self._get_fallback_explanation(prediction_result)
            
            # The rest of the function continues from here if we didn't return in the exception handlers
            
            # Just return the raw response without trying to parse it
            logger.info("Returning raw response from Bedrock")
            
            return {
                'explanation': completion,
                'source': 'AWS Bedrock'
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
        
        fallback_text = f"Information about {disease} is not available at the moment. Please consult with a healthcare professional for proper diagnosis and treatment."
        
        return {
            'explanation': fallback_text,
            'source': 'Fallback system (Bedrock unavailable)'
        }
        
    def chat(self, message):
        """
        Send a text message to Bedrock and get a response
        
        Args:
            message (str): The user's message
            
        Returns:
            dict: Raw response from Bedrock
        """
        try:
            logger.info(f"Sending message to Bedrock: {message[:100]}...")
            
            response = self.bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "messages": [
                        {"role": "user", "content": message}
                    ]
                }),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            logger.info("Successfully received response from Claude model")
            
            return {
                'success': True,
                'response': response_body
            }
            
        except Exception as e:
            logger.error(f"Error in Bedrock chat: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
