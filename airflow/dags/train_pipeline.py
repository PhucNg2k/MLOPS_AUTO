from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Param
import os
import json
import shutil
import subprocess
import mlflow
from airflow.utils.dates import days_ago
from dotenv import load_dotenv
import logging
import boto3
from airflow.hooks.S3_hook import S3Hook
from typing import List, Set

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add project paths
AIRFLOW_HOME = os.environ.get('AIRFLOW_HOME', '/opt/airflow')
PROJECT_ROOT = os.path.dirname(AIRFLOW_HOME)

# Initialize AWS connection
s3_hook = S3Hook(aws_conn_id='aws_default')

# Verify required environment variables
required_env_vars = [
    'TRAIN_BUCKET', 
    'VALIDATE_BUCKET', 
    'DVC_REMOTE_URL',
    'GITHUB_TOKEN',
    'GIT_USER_EMAIL',
    'GIT_USER_NAME',
    'GIT_REPO_URL',
    'GIT_BRANCH'
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Using TRAIN_BUCKET: {os.getenv('TRAIN_BUCKET')}")
logger.info(f"Using VALIDATE_BUCKET: {os.getenv('VALIDATE_BUCKET')}")
logger.info(f"Using DVC_REMOTE_URL: {os.getenv('DVC_REMOTE_URL')}")

# Set MLflow tracking URI from environment variable (if present)
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

def get_processed_partitions():
    """Get list of already processed partition folders"""
    processed_path = os.path.join(AIRFLOW_HOME, 'data', 'processed_partitions.json')
    if os.path.exists(processed_path):
        with open(processed_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_partition(partition):
    """Save partition as processed"""
    processed = get_processed_partitions()
    processed.add(partition)
    processed_path = os.path.join(AIRFLOW_HOME, 'data', 'processed_partitions.json')
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, 'w') as f:
        json.dump(list(processed), f)

def extract_partition_from_key(s3_key: str) -> str:
    """Extract partition folder name from S3 key."""
    # Example key: train/partition_1/image_labels.json
    parts = s3_key.strip('/').split('/')
    for i, part in enumerate(parts):
        if part.startswith('partition_'):
            return part
    raise ValueError(f"No partition folder found in key: {s3_key}")

# Default arguments for DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1
}

def download_partition(partition_path: str) -> str:
    """
    Download a partition from S3 to local storage
    Args:
        partition_path: Name of partition folder (e.g. 'partition_1')
    Returns:
        Local path to the downloaded partition
    """
    if not partition_path:
        raise ValueError("partition_path cannot be None")
    
    logger.info(f"Processing partition: {partition_path}")
    
    # Create local directory for partition
    local_path = os.path.join(AIRFLOW_HOME, 'data', 'training_data', partition_path)
    
    # Check if partition already exists locally
    if os.path.exists(local_path):
        logger.info(f"Partition already exists locally at {local_path}")
        return local_path
        
    # Construct S3 path
    s3_path = f"{os.getenv('TRAIN_BUCKET')}/{partition_path}"
    logger.info(f"Checking S3 partition structure: {s3_path}")
    
    try:
        # Use Airflow's S3 hook
        s3 = s3_hook.get_conn()
        
        # Parse bucket and prefix
        bucket_name = s3_path.split('/')[2]
        prefix = '/'.join(s3_path.split('/')[3:])
        
        # List all objects in partition
        response = s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            raise ValueError(f"No files found in S3 path: {s3_path}")
            
        # Check required structure
        found_items = {
            'images': False,
            'image_labels.json': False,
            'partition_stats.json': False
        }
        
        for obj in response['Contents']:
            key = obj['Key'].replace(prefix + '/', '')
            if key.startswith('images/'):
                found_items['images'] = True
            elif key == 'image_labels.json':
                found_items['image_labels.json'] = True
            elif key == 'partition_stats.json':
                found_items['partition_stats.json'] = True
                
        # Check if any required items are missing
        missing_items = [item for item, found in found_items.items() if not found]
        if missing_items:
            raise ValueError(f"Invalid partition structure in S3. Missing: {', '.join(missing_items)}")
            
        logger.info("S3 partition structure validated successfully")
        
        # Create directory for download
        os.makedirs(local_path, exist_ok=True)
            
        # Get AWS credentials from Airflow connection
        aws_creds = s3_hook.get_credentials()
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_creds.access_key
        env['AWS_SECRET_ACCESS_KEY'] = aws_creds.secret_key
        if aws_creds.token:
            env['AWS_SESSION_TOKEN'] = aws_creds.token
            
        logger.info(f"Downloading validated partition from S3: {s3_path}")
        # Use --recursive to ensure folder download, and --no-sign-request if public bucket (not needed here)
        result = subprocess.run([
            'aws', 's3', 'cp', s3_path, local_path, '--recursive'
        ], capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            logger.error(f"AWS CLI error (stdout): {result.stdout}")
            logger.error(f"AWS CLI error (stderr): {result.stderr}")
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            raise RuntimeError(f"Failed to download partition: aws s3 cp exited with {result.returncode}")
        
        logger.info(f"Successfully downloaded partition to {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Error processing partition: {str(e)}")
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        raise

def version_partition(partition_path: str, **context) -> None:
    """
    Version the partition using DVC. Handles both new partitions and updates to existing ones.
    
    NOTE: DVC push to remote is NOT performed here. Only local DVC tracking is done.
    DVC push is handled by GitHub Actions after .dvc files are committed and pushed to the repo.
    
    Args:
        partition_path: Local path to partition data (e.g. /opt/airflow/data/training_data/partition_1)
    """
    import subprocess
    
    partition_name = os.path.basename(partition_path)
    logger.info(f"Versioning partition with DVC: {partition_name}")
    
    try:
        # Initialize DVC if not already done
        if not os.path.exists('.dvc'):
            logger.info("Initializing DVC repository")
            subprocess.run(['dvc', 'init', '--no-scm'], check=True)
            
            # Configure remote storage (for local tracking only)
            dvc_remote = os.getenv('DVC_REMOTE_URL')
            if not dvc_remote:
                raise ValueError("DVC_REMOTE_URL environment variable must be set")
                
            logger.info("Configuring DVC remote storage (local only, no push)")
            subprocess.run(['dvc', 'remote', 'add', '--local', '-d', 'storage', dvc_remote], check=True)
            
            # Configure AWS credentials from Airflow connection
            aws_creds = s3_hook.get_credentials()
            subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'access_key_id', aws_creds.access_key], check=True)
            subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'secret_access_key', aws_creds.secret_key], check=True)
            if aws_creds.token:
                subprocess.run(['dvc', 'remote', 'modify', '--local', 'storage', 'session_token', aws_creds.token], check=True)
        
        # Check if partition is already tracked
        dvc_file = f"{partition_path}.dvc"
        is_update = os.path.exists(dvc_file)
        
        if is_update:
            logger.info(f"Partition {partition_name} already tracked, updating DVC hash")
            # DVC will automatically detect changes and update the hash
        else:
            logger.info(f"New partition {partition_name}, adding to DVC tracking")
        
        # Add/update partition in DVC
        subprocess.run(['dvc', 'add', partition_path], check=True)
        logger.info(f"DVC add complete for partition: {partition_name} (no push performed)")
        # NOTE: DVC push is intentionally omitted. See project docs and GitHub Actions workflow.
        # No hash extraction or extra logic needed; hash is visible in the .dvc file and in git history after merge.
        
        # Create/update metadata file for MLflow tracking
        metadata = {
            'timestamp': context['ts'],
            'partition': partition_name,
            'is_update': is_update
        }
        
        metadata_file = os.path.join(partition_path, 'dvc_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"{'Updated' if is_update else 'Created'} DVC metadata file: {metadata_file}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC operation failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during DVC versioning: {str(e)}")
        raise

def train_model(partition_path: str, **context) -> str:
    """
    Train model on a partition
    
    Args:
        partition_path: Local path to partition data
        
    Returns:
        str: Path to saved model
    """
    import subprocess
    from datetime import datetime
    import shutil

    # Set up model directories
    temp_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models')
    git_model_dir = os.path.join(AIRFLOW_HOME, 'models')
    os.makedirs(temp_model_dir, exist_ok=True)
    os.makedirs(git_model_dir, exist_ok=True)

    # Generate model filename with timestamp
    model_filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    temp_model_path = os.path.join(temp_model_dir, model_filename)
    final_model_path = os.path.join(git_model_dir, model_filename)

    # Start MLflow run
    with mlflow.start_run(run_name=f"train_{os.path.basename(partition_path)}"):
        # Run training script with correct arguments
        train_script = os.path.join(AIRFLOW_HOME, 'src', 'train.py')
        cmd = [
            'python', train_script,
            '--data_dir', partition_path,
            '--output_dir', temp_model_dir
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Training subprocess failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            raise

        # Log command output to MLflow
        mlflow.log_param("training_output", result.stdout)
        if result.stderr:
            mlflow.log_param("training_errors", result.stderr)

        # Copy model to git-tracked directory
        # (Assume best_model.pth is the output from train.py)
        best_model_path = os.path.join(temp_model_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, final_model_path)
        else:
            raise FileNotFoundError(f"Expected model file not found: {best_model_path}")

        # Setup git for commit
        git_email = os.getenv('GIT_USER_EMAIL')
        git_name = os.getenv('GIT_USER_NAME')
        git_branch = os.getenv('GIT_BRANCH', 'dev')

        logger.info(f"Configuring git with user {git_name}")
        subprocess.run(['git', 'config', '--global', 'user.email', git_email], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', git_name], check=True)

        # Configure git repository
        if os.getenv('GIT_REPO_URL'):
            logger.info("Setting git remote URL")
            try:
                subprocess.run(['git', 'remote', 'add', 'origin', os.getenv('GIT_REPO_URL')], check=True)
            except subprocess.CalledProcessError:
                subprocess.run(['git', 'remote', 'set-url', 'origin', os.getenv('GIT_REPO_URL')], check=True)

        # Commit new model
        subprocess.run(['git', 'add', final_model_path], check=True)
        commit_msg = f"Add new model trained on {os.path.basename(partition_path)}"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

        # Configure git credentials securely using git credential helper
        if os.getenv('GITHUB_TOKEN') and os.getenv('GIT_REPO_URL', '').startswith('https://'):
            git_cred_cmd = f"""
            echo -e "protocol=https\nhost=github.com\nusername=x-access-token\npassword={os.getenv('GITHUB_TOKEN')}" | git credential approve
            """
            subprocess.run(git_cred_cmd, shell=True, check=True)

        # Push to the specified branch
        logger.info(f"Pushing to {git_branch} branch")
        subprocess.run(['git', 'push', 'origin', git_branch], check=True)

    return final_model_path

def cleanup(**context):
    """
    Selective cleanup of temporary files while preserving important data.
    
    We keep:
    - Training data (for reproducibility)
    - Models (for versioning)
    - Evaluation results (for analysis and MLflow)
    
    We clean:
    - Temporary validation data (can be re-downloaded)
    - Any temporary files from training
    """
    # Only clean validation data as it can be re-downloaded
    validation_dir = os.path.join(AIRFLOW_HOME, 'data', 'validation_data')
    if os.path.exists(validation_dir):
        logger.info(f"Cleaning validation data directory: {validation_dir}")
        shutil.rmtree(validation_dir)
    
    # Clean any temporary training artifacts
    temp_model_dir = os.path.join(AIRFLOW_HOME, 'data', 'models')
    if os.path.exists(temp_model_dir):
        logger.info(f"Cleaning temporary model directory: {temp_model_dir}")
        shutil.rmtree(temp_model_dir)
    
    logger.info("Preserved directories for analysis:")
    logger.info(f"- Training data: {os.path.join(AIRFLOW_HOME, 'data', 'training_data')}")
    logger.info(f"- Final models: {os.path.join(AIRFLOW_HOME, 'models')}")
    logger.info(f"- Evaluation results: {os.path.join(AIRFLOW_HOME, 'data', 'evaluation_results')}")

with DAG(
    'animal_classification_training_v11',
    default_args=default_args,
    description='Train animal classification model on new partitions',
    schedule_interval='*/10 * * * *',  # Run every 5 minutes
    catchup=False
) as dag:

    # NOTE: DVC push to remote is handled by GitHub Actions, not by Airflow. Airflow only tracks data locally with DVC.
    # See project README and .github/workflows for details.

    # 1. List all partition folders in TRAIN_BUCKET
    def list_s3_partitions(**context):
        bucket_url = os.getenv('TRAIN_BUCKET')
        bucket_name = bucket_url.split('/')[2]
        prefix = '/'.join(bucket_url.split('/')[3:]).rstrip('/') + '/'
        s3 = s3_hook.get_conn()
        paginator = s3.get_paginator('list_objects_v2')
        partitions = set()
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
            # Use CommonPrefixes to get folder names
            for cp in page.get('CommonPrefixes', []):
                folder = cp['Prefix'][len(prefix):].rstrip('/')
                if folder.startswith('partition_'):
                    partitions.add(folder)
            # Fallback: also check keys for legacy/flat structure
            for obj in page.get('Contents', []):
                key = obj['Key']
                rel = key[len(prefix):]
                parts = rel.split('/')
                if parts[0].startswith('partition_'):
                    partitions.add(parts[0])
        logger.info(f"[DEBUG] list_s3_partitions: Found partitions: {partitions}")
        processed = get_processed_partitions()
        unprocessed = [p for p in sorted(partitions) if p not in processed]
        logger.info(f"[DEBUG] list_s3_partitions: Unprocessed partitions: {unprocessed}")
        if not unprocessed:
            logger.info("No new unprocessed partitions found. Skipping run.")
            return None
        selected_partition = unprocessed[0]
        logger.info(f"[DEBUG] list_s3_partitions: Selected partition: {selected_partition}")
        return selected_partition

    list_partitions_task = PythonOperator(
        task_id='list_s3_partitions',
        python_callable=list_s3_partitions,
        provide_context=True
    )

    # 2. Check if partition is already processed (redundant, but keeps logic modular)
    def check_partition_not_processed(partition, **context):
        if not partition:
            logger.info("No partition to process. Skipping downstream tasks.")
            return None
        processed = get_processed_partitions()
        if partition in processed:
            logger.info(f"Partition {partition} already processed. Skipping.")
            return None
        logger.info(f"[DEBUG] check_partition_not_processed: Passing partition '{partition}' to downstream task.")
        return partition

    check_partition_task = PythonOperator(
        task_id='check_partition_not_processed',
        python_callable=check_partition_not_processed,
        op_kwargs={
            'partition': "{{ task_instance.xcom_pull(task_ids='list_s3_partitions') }}"
        },
        provide_context=True
    )

    # 3. Download partition (now receives partition name)
    def download_partition_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to download. Skipping.")
            return None
        result = download_partition(partition_path)
        logger.info(f"[DEBUG] download_partition: Passing local path '{result}' to downstream task.")
        return result

    download_task = PythonOperator(
        task_id='download_partition',
        python_callable=download_partition_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='check_partition_not_processed') }}"
        }
    )

    # 4. Version with DVC
    def version_partition_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to version. Skipping.")
            return None
        version_partition(partition_path, **context)
        logger.info(f"[DEBUG] version_partition: Versioned partition at '{partition_path}'. Passing to downstream task.")
        return partition_path

    version_task = PythonOperator(
        task_id='version_partition',
        python_callable=version_partition_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='download_partition') }}"
        }
    )

    # 5. Train model
    def train_model_with_log(partition_path: str, **context):
        if not partition_path:
            logger.info("No partition to train on. Skipping.")
            return None
        model_path = train_model(partition_path, **context)
        logger.info(f"[DEBUG] train_model: Passing model path '{model_path}' to downstream task.")
        return model_path

    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_with_log,
        op_kwargs={
            'partition_path': "{{ task_instance.xcom_pull(task_ids='download_partition') }}"
        }
    )

    # 6. Cleanup (skip evaluation step)
    def cleanup_with_log(**context):
        cleanup(**context)
        logger.info(f"[DEBUG] cleanup: Cleanup complete. Passing None to downstream task.")
        return None

    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_with_log
    )

    # 7. Mark partition as processed
    def _mark_processed(**context):
        partition = context['task_instance'].xcom_pull(task_ids='check_partition_not_processed')
        if partition:
            save_processed_partition(partition)
        logger.info(f"[DEBUG] mark_processed: Marked partition '{partition}' as processed.")
        return partition

    mark_processed = PythonOperator(
        task_id='mark_processed',
        python_callable=_mark_processed
    )

    # Set task dependencies (no evaluation step)
    list_partitions_task >> check_partition_task >> download_task >> version_task >> train_task >> cleanup_task >> mark_processed