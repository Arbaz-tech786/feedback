"""
AWS Configuration Helper for Streamlit Cloud Deployment
This module provides robust AWS client initialization with proper error handling
and configuration management for both local and cloud environments.
"""

import os
import boto3
import json
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def get_aws_config() -> dict:
    """
    Get AWS configuration from environment variables or Streamlit secrets.
    Returns a dictionary with AWS configuration parameters.
    """
    config = {}
    
    # Try to get AWS credentials from environment variables first
    # These will be automatically picked up by boto3 if available
    config['aws_access_key_id'] = os.environ.get('AWS_ACCESS_KEY_ID')
    config['aws_secret_access_key'] = os.environ.get('AWS_SECRET_ACCESS_KEY')
    config['aws_default_region'] = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    
    # Log configuration status (without exposing secrets)
    if config['aws_access_key_id'] and config['aws_secret_access_key']:
        logger.info(f"Using AWS credentials from environment variables in region {config['aws_default_region']}")
    else:
        logger.info("No AWS credentials found in environment variables, using default credential chain")
    
    return config

def init_aws_clients_with_retry(max_retries: int = 3) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """
    Initialize AWS clients with retry logic and better error handling.
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (dynamodb, s3, bedrock_runtime) clients or None if initialization fails
    """
    aws_config = get_aws_config()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing AWS clients (attempt {attempt + 1}/{max_retries})")
            
            # Create session with explicit configuration if available
            if aws_config['aws_access_key_id'] and aws_config['aws_secret_access_key']:
                session = boto3.Session(
                    aws_access_key_id=aws_config['aws_access_key_id'],
                    aws_secret_access_key=aws_config['aws_secret_access_key'],
                    region_name=aws_config['aws_default_region']
                )
            else:
                # Use default credential chain (IAM role, environment variables, etc.)
                session = boto3.Session(region_name=aws_config['aws_default_region'])
            
            # Initialize clients
            dynamodb = session.resource('dynamodb')
            s3 = session.client('s3')
            bedrock_runtime = session.client('bedrock-runtime')
            
            # Test connections with simple operations
            logger.info("Testing AWS connections...")
            
            # Test DynamoDB
            try:
                dynamodb.meta.client.list_tables()
                logger.info("DynamoDB connection successful")
            except Exception as e:
                logger.warning(f"DynamoDB connection test failed: {str(e)}")
            
            # Test S3
            try:
                s3.list_buckets()
                logger.info("S3 connection successful")
            except Exception as e:
                logger.warning(f"S3 connection test failed: {str(e)}")
            
            # Test Bedrock
            try:
                bedrock_runtime.list_foundation_models()
                logger.info("Bedrock connection successful")
            except Exception as e:
                logger.warning(f"Bedrock connection test failed: {str(e)}")
            
            logger.info("AWS clients initialized successfully")
            return dynamodb, s3, bedrock_runtime
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached, AWS client initialization failed")
                return None, None, None
            continue
    
    return None, None, None

def check_aws_resources_exist(dynamodb, s3, bedrock_runtime, table_names: list, bucket_name: str) -> dict:
    """
    Check if required AWS resources exist and are accessible.
    
    Args:
        dynamodb: DynamoDB resource
        s3: S3 client
        bedrock_runtime: Bedrock runtime client
        table_names: List of DynamoDB table names to check
        bucket_name: S3 bucket name to check
        
    Returns:
        Dictionary with resource availability status
    """
    status = {
        'dynamodb_tables': {},
        's3_bucket': False,
        'bedrock_access': False,
        'all_resources_available': False
    }
    
    # Check DynamoDB tables
    if dynamodb:
        try:
            existing_tables = dynamodb.meta.client.list_tables()['TableNames']
            for table_name in table_names:
                if table_name in existing_tables:
                    status['dynamodb_tables'][table_name] = True
                    logger.info(f"DynamoDB table '{table_name}' found and accessible")
                else:
                    status['dynamodb_tables'][table_name] = False
                    logger.error(f"DynamoDB table '{table_name}' not found")
        except Exception as e:
            logger.error(f"Error checking DynamoDB tables: {str(e)}")
    
    # Check S3 bucket
    if s3:
        try:
            s3.head_bucket(Bucket=bucket_name)
            status['s3_bucket'] = True
            logger.info(f"S3 bucket '{bucket_name}' found and accessible")
        except Exception as e:
            logger.error(f"Error accessing S3 bucket '{bucket_name}': {str(e)}")
    
    # Check Bedrock access
    if bedrock_runtime:
        try:
            bedrock_runtime.list_foundation_models()
            status['bedrock_access'] = True
            logger.info("Bedrock access confirmed")
        except Exception as e:
            logger.error(f"Error accessing Bedrock: {str(e)}")
    
    # Overall status
    status['all_resources_available'] = (
        all(status['dynamodb_tables'].values()) and 
        status['s3_bucket'] and 
        status['bedrock_access']
    )
    
    return status

def get_resource_status_message(status: dict) -> str:
    """
    Generate a user-friendly message about resource availability.
    
    Args:
        status: Resource status dictionary from check_aws_resources_exist
        
    Returns:
        User-friendly status message
    """
    if status['all_resources_available']:
        return "✅ All AWS resources are accessible"
    
    messages = []
    
    # DynamoDB table status
    missing_tables = [name for name, exists in status['dynamodb_tables'].items() if not exists]
    if missing_tables:
        messages.append(f"❌ Missing DynamoDB tables: {', '.join(missing_tables)}")
    
    # S3 bucket status
    if not status['s3_bucket']:
        messages.append("❌ S3 bucket not accessible")
    
    # Bedrock status
    if not status['bedrock_access']:
        messages.append("❌ Bedrock access denied")
    
    return "\n".join(messages)