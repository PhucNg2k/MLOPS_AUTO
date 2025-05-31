# MLOps Animal Classification Project

An end-to-end MLOps pipeline for animal image classification using Airflow, DVC, PyTorch, and AWS S3. The pipeline automates data versioning, model training, evaluation, and deployment.

---

## Project Structure

```
.
├── airflow/                  # Airflow home directory
│   ├── dags/                # Airflow DAG definitions
│   ├── logs/                # Airflow logs
│   ├── plugins/             # Airflow plugins
│   └── config/              # Airflow configuration
├── src/                     # Source code (train.py, evaluate.py, model.py)
├── utils/                   # Utility scripts (DVC, data, upload, etc.)
├── data/                    # Data directory (raw, processed, dvc_store)
├── models/                  # Trained models
├── .dvc/                    # DVC configuration
├── .github/                 # GitHub Actions workflows
├── docker-compose.yaml      # Docker services configuration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Prerequisites
- Docker and Docker Compose
- AWS Account with S3 access
- Python 3.10 (for local development)
- Git (for version control)
- DVC (for data version control)

---

## Setup

### 1. Clone the repository
```powershell
git clone <repository-url>
cd <project-directory>
```

### 2. Create a `.env` file in the project root
```env
AIRFLOW_UID=50000
AIRFLOW_GID=0
AIRFLOW_HOME=/opt/airflow
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
BUCKET_NAME=your-bucket-name
TRAIN_BUCKET=s3://${BUCKET_NAME}/train
VALIDATE_BUCKET=s3://${BUCKET_NAME}/validate
DVC_REMOTE_URL=s3://${BUCKET_NAME}/dvc
GIT_USER_EMAIL=your_email@example.com
GIT_USER_NAME=your_github_username
GIT_REPO_URL=https://github.com/youruser/yourrepo
GIT_BRANCH=dev
GITHUB_TOKEN=your_github_token
PROJECT_DATA_DIR=./data
MODEL_STORE_DIR=./models
```

### 3. Create required directories (if not using Docker Compose volumes)
```powershell
mkdir airflow\dags airflow\logs airflow\plugins airflow\config
mkdir data\raw data\processed data\dvc_store
mkdir models
```

---

## Running the Project

### 1. Build and start all services
```powershell
docker compose build
```

### 2. Initialize Airflow and create admin user
```powershell
docker compose run airflow-init
```

### 3. Start all services in detached mode
```powershell
docker compose up -d
```

### 4. Access the Airflow web interface
- URL: http://localhost:8080
- Username: airflow (or as set in your .env)
- Password: airflow (or as set in your .env)

---

## Data Pipeline Usage

### 1. Download and partition the dataset
```powershell
python utils/download_animals10.py --output_dir data/raw
python utils/partition_dataset.py --source_dir data/raw --output_dir data/processed --num_partitions 4
```

### 2. Prepare and upload validation dataset
```powershell
python utils/prepare_validation_set.py --source_dir data/raw --bucket_path $env:VALIDATE_BUCKET
```

### 3. Upload training partitions to S3
```powershell
python utils/upload_partitions.py --partition_path data/processed/partition_1 --bucket_path $env:TRAIN_BUCKET
# Repeat for each partition
```

---

## Data Version Control (DVC)

```powershell
python utils/dvc_operations.py --operation init
python utils/dvc_operations.py --operation track --partition_dir data/processed/partition_1
python utils/dvc_operations.py --operation verify
```

---

## Training Pipeline (Airflow)

- The pipeline is event-driven: when a new partition is uploaded to S3, Airflow detects it and runs the pipeline.
- Steps:
  1. Detect new partition in S3
  2. Download and version with DVC
  3. Train the model
  4. Evaluate on validation set
  5. Push model to GitHub if accuracy threshold is met
- Monitor the pipeline via the Airflow UI at http://localhost:8080

---

## Troubleshooting

- **Permission Issues:**
  - Ensure `AIRFLOW_UID=50000` and `AIRFLOW_GID=0` in `.env` (no comments on the line)
  - Use `docker compose down --remove-orphans` if you see orphan container errors
- **AWS/DVC Issues:**
  - Check your AWS credentials and S3 bucket permissions
  - Run: `python utils/dvc_operations.py --operation verify`
- **General:**
  - Check logs: `docker compose logs -f`
  - Ensure all required directories exist
  - Ensure environment variables are set

---

## Development
- Place new DAGs in `airflow/dags/` (auto-detected by Airflow)
- Core ML code in `src/`
- Utilities in `utils/`
- Data in `data/`, models in `models/`

---

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License
[Your License]