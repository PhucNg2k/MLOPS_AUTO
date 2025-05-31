# MLOps Animal Classification Project

An end-to-end MLOps pipeline for animal image classification using Airflow, DVC, and PyTorch. The pipeline includes automated data versioning, model training, evaluation, and deployment.

## Project Structure

```
.
├── airflow/                  # Airflow home directory
│   ├── dags/                # Airflow DAG definitions
│   ├── logs/                # Airflow logs
│   ├── plugins/             # Airflow plugins
│   └── config/              # Airflow configuration
├── src/                     # Source code
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── model.py            # Model architecture
├── utils/                   # Utility scripts
│   ├── dvc_operations.py   # DVC management
│   ├── upload_partitions.py # Data upload utilities
│   ├── partition_dataset.py # Dataset partitioning
│   ├── prepare_validation_set.py # Validation set preparation
│   └── download_animals10.py # Dataset download
├── data/                    # Data directory
│   ├── raw/                # Original downloads
│   ├── processed/          # Processed datasets
│   └── dvc_store/          # DVC cache
├── models/                  # Trained models
├── .dvc/                   # DVC configuration
├── .github/                # GitHub Actions workflows
├── docker-compose.yaml     # Docker services configuration
├── Dockerfile              # Docker image definition
└── requirements.txt        # Python dependencies
```

## Prerequisites

- Docker and Docker Compose
- AWS Account with S3 access
- Python 3.10 (for local development)
- Git (for version control)
- DVC (for data version control)

## Initial Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a `.env` file in the project root:
```bash
# Airflow settings
AIRFLOW_UID=50000  # On Linux, use: $(id -u)
AIRFLOW_HOME=/path/to/your/airflow

# AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region  # e.g., us-east-1

# S3 Bucket Configuration
BUCKET_NAME=your-bucket-name
TRAIN_BUCKET=${BUCKET_NAME}/train
VALIDATE_BUCKET=${BUCKET_NAME}/validate

# DVC Remote Storage
DVC_REMOTE_URL=s3://${BUCKET_NAME}/dvc

# Project paths
PROJECT_DATA_DIR=./data  # Local project data directory
MODEL_STORE_DIR=./models  # Local project models directory
```

3. Create required directories:
```bash
# Create Airflow directories
mkdir -p airflow/{dags,logs,plugins,config}

# Create project directories
mkdir -p data/{raw,processed,dvc_store}
mkdir -p models
```

## Running the Project

1. Build and start the services:
```bash
docker compose up --build
```

2. In a new terminal, initialize Airflow and create admin user:
```bash
docker compose run airflow-init
```

3. Start all services in detached mode:
```bash
docker compose up -d
```

4. Access the Airflow web interface:
- URL: http://localhost:8080
- Username: admin
- Password: admin

## Data Pipeline Usage

1. Download and partition the dataset:
```bash
# Download Animals-10 dataset
python utils/download_animals10.py --output_dir data/raw

# Create partitions
python utils/partition_dataset.py \
    --source_dir data/raw \
    --output_dir data/processed \
    --num_partitions 4
```

2. Prepare validation dataset:
```bash
python utils/prepare_validation_set.py \
    --source_dir data/raw \
    --bucket_path $VALIDATE_BUCKET
```

3. Upload training partitions:
```bash
python utils/upload_partitions.py \
    --partition_path data/processed/partition_1 \
    --bucket_path $TRAIN_BUCKET
```

## Data Version Control

DVC is used to track data and model versions:

```bash
# Initialize DVC
python utils/dvc_operations.py --operation init

# Track new data
python utils/dvc_operations.py --operation track --partition_dir data/processed/partition_1

# Verify DVC setup
python utils/dvc_operations.py --operation verify
```

## Training Pipeline

The training pipeline is automated through Airflow and includes:
1. Detection of new training partitions in S3
2. Downloading and versioning data with DVC
3. Training the model
4. Evaluation against validation set
5. Model promotion if accuracy threshold is met

Monitor the pipeline through the Airflow UI at http://localhost:8080.

## Development

### Adding New DAGs
Place DAG files in `airflow/dags/`. They will be automatically picked up by Airflow.

### Local Development
For local development without Docker:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### Code Structure
- `src/`: Core ML code (training, evaluation, model definition)
- `utils/`: Support functions (data processing, DVC operations)
- `airflow/dags/`: Workflow definitions
- `models/`: Saved model checkpoints
- `data/`: Dataset storage

## Troubleshooting

1. Permission Issues:
   - Set correct AIRFLOW_UID in .env
   - Ensure proper file permissions
   - Check Docker volume mount permissions

2. Docker Issues:
   - Check logs: `docker compose logs -f`
   - Verify all required directories exist
   - Ensure environment variables are set

3. DVC Issues:
   - Verify AWS credentials
   - Check S3 bucket permissions
   - Run: `python utils/dvc_operations.py --operation verify`

4. Data Pipeline Issues:
   - Check S3 bucket access
   - Verify data directory structure
   - Ensure correct environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License]

## Path Configuration

The project uses two types of paths:

1. Local Project Paths (on your machine):
```
project_root/
├── data/          # PROJECT_DATA_DIR (./data)
└── models/        # MODEL_STORE_DIR (./models)
```

2. Docker Container Paths (automatically mapped):
```
/opt/airflow/
├── data/          # Mapped from local PROJECT_DATA_DIR
└── models/        # Mapped from local MODEL_STORE_DIR
```

The docker-compose.yaml handles the mapping between your local paths and the container paths automatically. When setting up the `.env` file, use the local paths (starting with `./`). The Docker container will map these to the appropriate locations in `/opt/airflow/`. 