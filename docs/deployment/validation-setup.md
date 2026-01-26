# Startup Validation System - Operator Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration Guide](#configuration-guide)
3. [Running Validation](#running-validation)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [CI/CD Integration](#cicd-integration)
6. [Docker/Kubernetes Deployment](#dockerkubernetes-deployment)
7. [Performance Expectations](#performance-expectations)
8. [Security Best Practices](#security-best-practices)
9. [Validation Output Examples](#validation-output-examples)
10. [FAQ](#faq)

---

## Overview

The startup validation system is a critical safety mechanism that prevents broken deployments from reaching production. It validates essential dependencies (ML models and database connectivity) before the API accepts traffic, ensuring that deployments fail fast if critical components are missing or misconfigured.

### Why It's Important

Without validation, a deployment with missing models or database connection issues would:
- Start successfully but fail on first request
- Create a poor user experience
- Require manual intervention to diagnose
- Potentially cause data inconsistencies

The validation system catches these issues **before** the API starts accepting requests, making deployments more reliable and easier to troubleshoot.

### How It Works

The validation system operates at two levels:

1. **Standalone Pre-Deployment Validation** (`scripts/validate_deployment.py`):
   - Can be run independently by operators before deployment
   - Validates models and database without starting the API
   - Returns exit code 0 (success) or 1 (failure) for CI/CD integration
   - Provides detailed error messages and fix instructions

2. **Automatic Startup Validation** (integrated in `main.py`):
   - Runs automatically when the FastAPI application starts
   - Prevents API startup if validations fail
   - Exits with code 1 if any validation fails
   - Logs detailed validation reports

### What Gets Validated

- **ML Models**: 
  - Model files exist and are readable
  - Optional: SHA-256 checksum verification for integrity
  - File size validation (non-empty files)
  
- **Database**:
  - Connection can be established
  - Basic query execution works
  - Connection timeout monitoring

---

## Configuration Guide

### Environment Variables

The validation system uses environment variables for configuration. These can be set in:
- `.env` file (for local development)
- Shell environment (for CI/CD)
- Container environment variables (for Docker/Kubernetes)
- Configuration file (JSON format, passed via `--config` flag)

#### Required Configuration

**`DATABASE_URL`** (Required)
- Database connection string
- Format: `postgresql://[user]:[password]@[host]:[port]/[database]`
- Example: `postgresql://fpl_user:password@db:5432/fpl_db`
- For Docker: Use service name (e.g., `db`) as hostname
- For local: Use `localhost` as hostname

#### Optional Configuration

**`MODEL_PATHS`** (Optional)
- Comma-separated list of model file paths
- Example: `/app/models/model1.pkl,/app/models/model2.pkl`
- If not provided, the system auto-detects models using PLEngine
- Default: `None` (auto-detect)

**`MODEL_CHECKSUMS`** (Optional)
- Model file checksums for integrity verification
- Format: `path1:checksum1,path2:checksum2`
- Checksums are SHA-256 hex strings
- Example: `/app/models/model1.pkl:abc123...,/app/models/model2.pkl:def456...`
- Default: `None` (no checksum verification)

**`DB_VALIDATION_TIMEOUT`** (Optional)
- Database connection timeout in seconds
- Default: `2.0`
- Increase for slow networks or high-latency databases
- Example: `5.0` for 5 seconds

**`MODEL_VALIDATION_TIMEOUT`** (Optional)
- Model file operation timeout in seconds
- Useful for network-mounted filesystems (NFS, EFS, etc.)
- Default: `None` (no timeout)
- Example: `30.0` for 30 seconds

**`VALIDATION_PERFORMANCE_BUDGET`** (Optional)
- Maximum time allowed for all validations during startup
- Default: `5.0` seconds
- Used only for startup validation (not standalone script)
- Example: `10.0` for 10 seconds

### Configuration File Format

You can also use a JSON configuration file instead of environment variables:

```json
{
  "model_paths": ["/app/models/model1.pkl", "/app/models/model2.pkl"],
  "model_checksums": {
    "/app/models/model1.pkl": "abc123...",
    "/app/models/model2.pkl": "def456..."
  },
  "database_url": "postgresql://fpl_user:password@db:5432/fpl_db",
  "db_timeout": 2.0,
  "model_timeout": 30.0
}
```

Save this as `validation-config.json` and pass it with `--config validation-config.json`.

### Configuration Priority

Configuration is loaded in the following priority order (highest to lowest):

1. **Command-line arguments** (highest priority)
2. **Configuration file** (if `--config` provided)
3. **Environment variables**
4. **Default values** (lowest priority)

### Example Configurations

#### Local Development

```bash
# .env file
DATABASE_URL=postgresql://fpl_user:password@localhost:5432/fpl_db
MODEL_PATHS=/app/models/attack_model.pkl,/app/models/defense_model.pkl
DB_VALIDATION_TIMEOUT=2.0
```

#### Docker Compose

```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - DATABASE_URL=postgresql://fpl_user:password@db:5432/fpl_db
      - MODEL_PATHS=/app/models/attack_model.pkl,/app/models/defense_model.pkl
      - DB_VALIDATION_TIMEOUT=2.0
```

#### Production (with checksums)

```bash
# Production .env
DATABASE_URL=postgresql://fpl_user:secure_password@prod-db.example.com:5432/fpl_db
MODEL_PATHS=/app/models/attack_model.pkl,/app/models/defense_model.pkl
MODEL_CHECKSUMS=/app/models/attack_model.pkl:abc123def456...,/app/models/defense_model.pkl:789ghi012jkl...
DB_VALIDATION_TIMEOUT=5.0
MODEL_VALIDATION_TIMEOUT=30.0
VALIDATION_PERFORMANCE_BUDGET=10.0
```

---

## Running Validation

### Standalone Pre-Deployment Validation

Run the validation script **before** deploying to catch issues early:

#### Basic Usage

```bash
# Using Docker Compose (recommended)
docker compose exec backend python3 scripts/validate_deployment.py
```

#### With Verbose Output

```bash
docker compose exec backend python3 scripts/validate_deployment.py --verbose
```

#### With Custom Configuration

```bash
# Using config file
docker compose exec backend python3 scripts/validate_deployment.py --config /app/config/validation-config.json

# Using command-line arguments
docker compose exec backend python3 scripts/validate_deployment.py \
  --model-paths /app/models/model1.pkl /app/models/model2.pkl \
  --database-url postgresql://user:pass@host:5432/dbname \
  --db-timeout 5.0
```

#### Exit Codes

- **0**: All validations passed - deployment environment is ready
- **1**: One or more validations failed - deployment should not proceed
- **130**: Validation interrupted by user (Ctrl+C)

### Automatic Startup Validation

The validation runs automatically when the FastAPI application starts. No manual action required, but you should monitor logs for validation results.

#### Checking Validation Results

```bash
# View startup logs
docker compose logs backend | grep -A 20 "STARTUP VALIDATION"

# Or follow logs in real-time
docker compose logs -f backend
```

#### What Happens on Failure

If startup validation fails:
1. Detailed error report is logged
2. API startup is aborted
3. Process exits with code 1
4. Container/process stops (prevents broken deployment)

---

## Troubleshooting Guide

### Common Validation Failures

#### 1. Model File Not Found

**Error Message:**
```
✗ ML Models: FAIL
  Status: UNHEALTHY
  Error: /app/models/model1.pkl: Model file not found
```

**Possible Causes:**
- Model files haven't been trained yet
- Model files are in a different location
- File permissions prevent reading
- Path is incorrect

**Solutions:**

1. **Train models first:**
   ```bash
   docker compose exec backend python3 scripts/train_ml_models.py
   ```

2. **Specify correct model paths:**
   ```bash
   # Check where models are located
   docker compose exec backend find /app -name "*.pkl" -type f
   
   # Then specify correct paths
   export MODEL_PATHS=/app/models/actual_model.pkl
   ```

3. **Check file permissions:**
   ```bash
   docker compose exec backend ls -la /app/models/
   # Ensure files are readable: chmod 644 /app/models/*.pkl
   ```

4. **Use auto-detection:**
   ```bash
   # Remove MODEL_PATHS to let system auto-detect
   unset MODEL_PATHS
   ```

#### 2. Checksum Mismatch

**Error Message:**
```
✗ ML Models: FAIL
  Status: UNHEALTHY
  Error: [checksum_mismatch] /app/models/model1.pkl: Checksum mismatch
```

**Possible Causes:**
- Model file was modified or corrupted
- Expected checksum in configuration is incorrect
- File was replaced with a different version

**Solutions:**

1. **Re-train the model:**
   ```bash
   docker compose exec backend python3 scripts/train_ml_models.py
   ```

2. **Update expected checksum:**
   ```bash
   # Calculate actual checksum
   docker compose exec backend python3 -c "
   import hashlib
   with open('/app/models/model1.pkl', 'rb') as f:
       print(hashlib.sha256(f.read()).hexdigest())
   "
   
   # Update MODEL_CHECKSUMS with new checksum
   export MODEL_CHECKSUMS=/app/models/model1.pkl:NEW_CHECKSUM_HERE
   ```

3. **Remove checksum verification (not recommended for production):**
   ```bash
   unset MODEL_CHECKSUMS
   ```

#### 3. Database Connection Refused

**Error Message:**
```
✗ Database: FAIL
  Status: UNHEALTHY
  Error: Error type: connection_refused | Database connection refused
```

**Possible Causes:**
- Database service is not running
- Database host/port is incorrect
- Network connectivity issues
- Firewall blocking connection

**Solutions:**

1. **Start database service:**
   ```bash
   docker compose up -d db
   # Wait for database to be ready
   docker compose exec db pg_isready -U fpl_user
   ```

2. **Verify DATABASE_URL:**
   ```bash
   # Check current DATABASE_URL
   docker compose exec backend env | grep DATABASE_URL
   
   # For Docker Compose, use service name 'db' as hostname
   export DATABASE_URL=postgresql://fpl_user:password@db:5432/fpl_db
   
   # For local connection, use 'localhost'
   export DATABASE_URL=postgresql://fpl_user:password@localhost:5432/fpl_db
   ```

3. **Test database connectivity:**
   ```bash
   # Test connection from backend container
   docker compose exec backend python3 -c "
   from sqlalchemy import create_engine, text
   import os
   engine = create_engine(os.getenv('DATABASE_URL'))
   with engine.connect() as conn:
       result = conn.execute(text('SELECT 1'))
       print('Connection OK:', result.fetchone())
   "
   ```

4. **Check network connectivity:**
   ```bash
   # From backend container, test if database is reachable
   docker compose exec backend ping -c 3 db
   docker compose exec backend nc -zv db 5432
   ```

#### 4. Database Authentication Failure

**Error Message:**
```
✗ Database: FAIL
  Status: UNHEALTHY
  Error: Error type: authentication_failure | Database authentication failed
```

**Possible Causes:**
- Incorrect username or password
- User doesn't exist in database
- Password has special characters that need escaping

**Solutions:**

1. **Verify credentials:**
   ```bash
   # Check .env file or environment variables
   docker compose exec backend env | grep DATABASE_URL
   
   # Test with psql directly
   docker compose exec db psql -U fpl_user -d fpl_db -c "SELECT 1"
   ```

2. **Reset database password:**
   ```bash
   # Connect as postgres superuser
   docker compose exec db psql -U postgres -c "ALTER USER fpl_user WITH PASSWORD 'new_password';"
   
   # Update DATABASE_URL
   export DATABASE_URL=postgresql://fpl_user:new_password@db:5432/fpl_db
   ```

3. **Escape special characters in password:**
   ```bash
   # If password contains special characters, URL-encode them
   # Example: password "p@ss#word" becomes "p%40ss%23word"
   export DATABASE_URL=postgresql://fpl_user:p%40ss%23word@db:5432/fpl_db
   ```

#### 5. Database Not Found

**Error Message:**
```
✗ Database: FAIL
  Status: UNHEALTHY
  Error: Error type: database_not_found | Database not found
```

**Possible Causes:**
- Database name in DATABASE_URL is incorrect
- Database hasn't been created yet

**Solutions:**

1. **Create the database:**
   ```bash
   docker compose exec db psql -U fpl_user -c "CREATE DATABASE fpl_db;"
   ```

2. **Verify database name:**
   ```bash
   # List all databases
   docker compose exec db psql -U fpl_user -l
   
   # Update DATABASE_URL with correct database name
   export DATABASE_URL=postgresql://fpl_user:password@db:5432/correct_db_name
   ```

3. **Run database initialization:**
   ```bash
   docker compose exec backend python3 scripts/init_database.py
   ```

#### 6. Connection Timeout

**Error Message:**
```
✗ Database: FAIL
  Status: UNHEALTHY
  Error: Error type: timeout | Connection timeout after 2.0s
```

**Possible Causes:**
- Database is slow to respond
- Network latency is high
- Database is overloaded
- Timeout value is too low

**Solutions:**

1. **Increase timeout:**
   ```bash
   export DB_VALIDATION_TIMEOUT=5.0
   # Or use command-line flag
   docker compose exec backend python3 scripts/validate_deployment.py --db-timeout 5.0
   ```

2. **Check database performance:**
   ```bash
   # Check database load
   docker compose exec db psql -U fpl_user -d fpl_db -c "SELECT * FROM pg_stat_activity;"
   ```

3. **Verify network latency:**
   ```bash
   docker compose exec backend ping -c 5 db
   ```

### Decision Tree for Common Issues

```
Validation Failed
│
├─ Model Validation Failed
│  │
│  ├─ "Model file not found"
│  │  ├─ Check if models exist: find /app -name "*.pkl"
│  │  ├─ Train models: python3 scripts/train_ml_models.py
│  │  └─ Verify MODEL_PATHS environment variable
│  │
│  ├─ "Checksum mismatch"
│  │  ├─ Re-train models
│  │  ├─ Update MODEL_CHECKSUMS with new checksum
│  │  └─ Or remove checksum verification (not recommended)
│  │
│  └─ "Permission denied" or "Unreadable"
│     └─ Fix file permissions: chmod 644 /app/models/*.pkl
│
└─ Database Validation Failed
   │
   ├─ "Connection refused"
   │  ├─ Start database: docker compose up -d db
   │  ├─ Verify DATABASE_URL (use 'db' as hostname in Docker)
   │  └─ Check network: ping db, nc -zv db 5432
   │
   ├─ "Authentication failed"
   │  ├─ Verify username/password in DATABASE_URL
   │  ├─ Test with psql directly
   │  └─ Reset password if needed
   │
   ├─ "Database not found"
   │  ├─ Create database: CREATE DATABASE fpl_db;
   │  └─ Verify database name in DATABASE_URL
   │
   └─ "Timeout"
      ├─ Increase DB_VALIDATION_TIMEOUT
      ├─ Check database performance
      └─ Verify network latency
```

---

## CI/CD Integration

The validation script is designed to integrate seamlessly with CI/CD pipelines. It returns standard exit codes (0 for success, 1 for failure) and provides structured output.

### GitHub Actions

#### Basic Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Compose
        run: |
          docker compose up -d db
          sleep 10  # Wait for database to be ready
      
      - name: Run pre-deployment validation
        run: |
          docker compose exec -T backend python3 scripts/validate_deployment.py --verbose
        env:
          DATABASE_URL: postgresql://fpl_user:password@db:5432/fpl_db
      
      - name: Deploy (only if validation passes)
        if: success()
        run: |
          # Your deployment steps here
```

#### Advanced Integration with Model Checksums

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Compose
        run: |
          docker compose up -d db
          sleep 10
      
      - name: Train ML models
        run: |
          docker compose exec -T backend python3 scripts/train_ml_models.py
      
      - name: Calculate model checksums
        id: checksums
        run: |
          CHECKSUMS=$(docker compose exec -T backend python3 -c "
          import hashlib
          import os
          models = ['/app/models/model1.pkl', '/app/models/model2.pkl']
          checksums = []
          for model in models:
              if os.path.exists(model):
                  with open(model, 'rb') as f:
                      checksum = hashlib.sha256(f.read()).hexdigest()
                  checksums.append(f'{model}:{checksum}')
          print(','.join(checksums))
          ")
          echo "checksums=$CHECKSUMS" >> $GITHUB_OUTPUT
      
      - name: Run pre-deployment validation
        run: |
          docker compose exec -T backend python3 scripts/validate_deployment.py --verbose
        env:
          DATABASE_URL: postgresql://fpl_user:${{ secrets.DB_PASSWORD }}@db:5432/fpl_db
          MODEL_CHECKSUMS: ${{ steps.checksums.outputs.checksums }}
          DB_VALIDATION_TIMEOUT: 5.0
      
      - name: Deploy
        if: success()
        run: |
          # Your deployment steps here
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - deploy

validate:
  stage: validate
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker compose up -d db
    - sleep 10
  script:
    - docker compose exec -T backend python3 scripts/validate_deployment.py --verbose
  variables:
    DATABASE_URL: "postgresql://fpl_user:${DB_PASSWORD}@db:5432/fpl_db"
  only:
    - main

deploy:
  stage: deploy
  image: docker:latest
  services:
    - docker:dind
  script:
    - echo "Deploying..."
    # Your deployment steps here
  only:
    - main
  when: on_success
```

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DATABASE_URL = 'postgresql://fpl_user:password@db:5432/fpl_db'
    }
    
    stages {
        stage('Validate') {
            steps {
                sh '''
                    docker compose up -d db
                    sleep 10
                    docker compose exec -T backend python3 scripts/validate_deployment.py --verbose
                '''
            }
        }
        
        stage('Deploy') {
            when {
                expression { currentBuild.result == 'SUCCESS' }
            }
            steps {
                echo 'Deploying...'
                // Your deployment steps here
            }
        }
    }
    
    post {
        always {
            sh 'docker compose down'
        }
    }
}
```

### Best Practices for CI/CD

1. **Run validation before deployment**: Always validate before deploying to catch issues early
2. **Use secrets for credentials**: Never hardcode passwords in CI/CD files
3. **Set appropriate timeouts**: Increase timeouts for CI environments if needed
4. **Capture validation output**: Save validation logs as artifacts for debugging
5. **Fail fast**: Let validation failures stop the pipeline immediately

---

## Docker/Kubernetes Deployment

### Docker Compose

#### Basic Setup

```yaml
# docker-compose.yml
services:
  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: fpl_user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: fpl_db
    volumes:
      - db_data:/var/lib/postgresql/data
  
  backend:
    build: ./backend
    depends_on:
      - db
    environment:
      DATABASE_URL: postgresql://fpl_user:password@db:5432/fpl_db
      MODEL_PATHS: /app/models/attack_model.pkl,/app/models/defense_model.pkl
      DB_VALIDATION_TIMEOUT: 2.0
    volumes:
      - ./backend/models:/app/models  # Mount model directory
    command: >
      sh -c "
        python3 scripts/validate_deployment.py &&
        uvicorn app.main:app --host 0.0.0.0 --port 8000
      "

volumes:
  db_data:
```

#### Pre-Startup Validation

```yaml
# docker-compose.yml
services:
  backend:
    # ... other config ...
    command: >
      sh -c "
        echo 'Running pre-deployment validation...' &&
        python3 scripts/validate_deployment.py --verbose &&
        echo 'Validation passed. Starting API...' &&
        uvicorn app.main:app --host 0.0.0.0 --port 8000
      "
```

**Note**: The validation also runs automatically in `main.py` startup event, so the explicit pre-startup validation is optional but recommended for clearer error messages.

### Kubernetes

#### ConfigMap for Configuration

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: validation-config
data:
  DB_VALIDATION_TIMEOUT: "5.0"
  MODEL_VALIDATION_TIMEOUT: "30.0"
  VALIDATION_PERFORMANCE_BUDGET: "10.0"
```

#### Secret for Database Credentials

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
type: Opaque
stringData:
  DATABASE_URL: postgresql://fpl_user:password@postgres-service:5432/fpl_db
```

#### Deployment with Init Container

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fpl-backend
spec:
  replicas: 3
  template:
    spec:
      initContainers:
      - name: validate-deployment
        image: fpl-backend:latest
        command: ["python3", "scripts/validate_deployment.py", "--verbose"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DATABASE_URL
        - name: DB_VALIDATION_TIMEOUT
          valueFrom:
            configMapKeyRef:
              name: validation-config
              key: DB_VALIDATION_TIMEOUT
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      containers:
      - name: backend
        image: fpl-backend:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: DATABASE_URL
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-files-pvc
```

#### Model Files in Kubernetes

**Option 1: Persistent Volume**

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-files-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Gi
```

**Option 2: ConfigMap (for small models)**

```yaml
# Not recommended for large model files, but possible for small models
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-files
binaryData:
  model1.pkl: <base64-encoded-model-file>
```

**Option 3: Init Container to Download Models**

```yaml
initContainers:
- name: download-models
  image: fpl-backend:latest
  command: ["sh", "-c", "python3 scripts/train_ml_models.py"]
  volumeMounts:
  - name: models
    mountPath: /app/models
```

### Container Considerations

1. **Model File Location**: Mount model files as read-only volumes
2. **Database Hostname**: Use Kubernetes service names (e.g., `postgres-service`) instead of `localhost`
3. **Network Timeouts**: Increase timeouts for network-mounted filesystems (NFS, EFS)
4. **Resource Limits**: Ensure containers have enough resources for validation
5. **Health Checks**: Use validation script for Kubernetes liveness/readiness probes

---

## Performance Expectations

### Typical Validation Times

- **Model Validation**: 0.1 - 0.5 seconds
  - File existence check: < 0.1s
  - Checksum calculation: 0.1 - 0.5s (depends on file size)
  
- **Database Validation**: 0.1 - 2.0 seconds
  - Connection establishment: 0.1 - 1.0s
  - Query execution: < 0.1s
  - Total: Usually < 2.0s with default timeout

- **Total Validation Time**: 0.2 - 2.5 seconds (typically < 1.0s)

### Performance Budget

The startup validation has a default performance budget of **5.0 seconds**. If validations exceed this budget, a warning is logged but validation continues.

**When to Increase Budget:**
- Network-mounted filesystems (NFS, EFS) with high latency
- Slow database connections
- Large model files requiring longer checksum calculation

**Configuration:**
```bash
export VALIDATION_PERFORMANCE_BUDGET=10.0  # 10 seconds
```

### Timeout Configuration

**Database Timeout** (`DB_VALIDATION_TIMEOUT`):
- Default: 2.0 seconds
- Recommended: 2.0 - 5.0 seconds
- For slow networks: 5.0 - 10.0 seconds

**Model Timeout** (`MODEL_VALIDATION_TIMEOUT`):
- Default: None (no timeout)
- For network filesystems: 30.0 - 60.0 seconds
- For local filesystems: Not needed (leave as None)

### Performance Monitoring

Validation times are logged during startup:

```
INFO: Model validation completed in 0.15s: healthy - OK
INFO: Database validation completed in 0.23s: healthy - OK
INFO: All validations completed within budget: 0.38s <= 5.0s
```

Monitor these logs to identify performance issues.

---

## Security Best Practices

### Credential Management

1. **Never commit credentials to version control**
   - Use `.env` files (in `.gitignore`)
   - Use secrets management systems (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Use CI/CD secret variables

2. **Use environment variables for credentials**
   ```bash
   # Good: Environment variable
   export DATABASE_URL=postgresql://user:password@host:5432/db
   
   # Bad: Hardcoded in code or config files
   DATABASE_URL=postgresql://user:password@host:5432/db  # In committed file
   ```

3. **Mask credentials in logs**
   - The validation system automatically masks passwords in database URLs
   - Example: `postgresql://user:***@host:5432/db`

### Model File Security

1. **File Permissions**
   ```bash
   # Restrict model file access
   chmod 644 /app/models/*.pkl  # Read-only for owner, read for group/others
   chown app_user:app_group /app/models/*.pkl
   ```

2. **Checksum Verification**
   - Use checksums in production to detect tampering
   - Store checksums securely (not in version control)
   - Rotate checksums when models are updated

3. **Secure Storage**
   - Store models in secure, access-controlled locations
   - Use encrypted volumes for sensitive models
   - Implement access logging for model files

### Network Security

1. **Database Connection**
   - Use TLS/SSL for database connections in production
   - Restrict database access to specific IPs/networks
   - Use strong passwords and rotate them regularly

2. **Internal Networks**
   - Use private networks for database communication
   - Avoid exposing database ports publicly
   - Use service meshes or network policies in Kubernetes

### Configuration Security

1. **Separate Configurations by Environment**
   - Development: Local database, no checksums
   - Staging: Staging database, optional checksums
   - Production: Production database, mandatory checksums

2. **Least Privilege**
   - Database user should have minimal required permissions
   - Application should not require superuser access

3. **Audit Logging**
   - Log all validation attempts (success and failure)
   - Monitor for suspicious validation failures
   - Alert on repeated validation failures

---

## Validation Output Examples

### Successful Validation

#### Standalone Script Output

```
2025-01-25 21:14:32 - __main__ - INFO - Starting pre-deployment validation...

======================================================================
PRE-DEPLOYMENT VALIDATION REPORT
======================================================================

✓ ML Models: PASS
  Status: HEALTHY
  Details: All checks passed

✓ Database: PASS
  Status: HEALTHY
  Details: All checks passed

----------------------------------------------------------------------
✓ OVERALL STATUS: READY FOR DEPLOYMENT
  All critical dependencies are validated and ready.
======================================================================

2025-01-25 21:14:32 - __main__ - INFO - ✓ All validations passed. Deployment environment is ready.
```

#### Startup Validation Output (Logs)

```
2025-01-25 21:14:32 - app.main - INFO - Starting startup health validation orchestrator...
2025-01-25 21:14:32 - app.services.startup_validation - INFO - Starting model validation...
2025-01-25 21:14:32 - app.services.startup_validation - INFO - Model validation completed in 0.15s: healthy - OK
2025-01-25 21:14:32 - app.services.startup_validation - INFO - Starting database validation...
2025-01-25 21:14:32 - app.services.startup_validation - INFO - Database validation completed in 0.23s: healthy - OK
2025-01-25 21:14:32 - app.services.startup_validation - INFO - All validations completed within budget: 0.38s <= 5.0s
2025-01-25 21:14:32 - app.main - INFO - ============================================================
2025-01-25 21:14:32 - app.main - INFO - STARTUP VALIDATION PASSED
2025-01-25 21:14:32 - app.main - INFO - ============================================================
2025-01-25 21:14:32 - app.main - INFO - Startup Validation Report
2025-01-25 21:14:32 - app.main - INFO - ============================================================
2025-01-25 21:14:32 - app.main - INFO - Timestamp: 2025-01-25T21:14:32.123456
2025-01-25 21:14:32 - app.main - INFO - 
2025-01-25 21:14:32 - app.main - INFO - ✓ ML Models: HEALTHY
2025-01-25 21:14:32 - app.main - INFO -   Timestamp: 2025-01-25T21:14:32.100000
2025-01-25 21:14:32 - app.main - INFO -   Status: OK
2025-01-25 21:14:32 - app.main - INFO - 
2025-01-25 21:14:32 - app.main - INFO - ✓ Database: HEALTHY
2025-01-25 21:14:32 - app.main - INFO -   Timestamp: 2025-01-25T21:14:32.200000
2025-01-25 21:14:32 - app.main - INFO -   Status: OK
2025-01-25 21:14:32 - app.main - INFO - 
2025-01-25 21:14:32 - app.main - INFO - ============================================================
2025-01-25 21:14:32 - app.main - INFO - Overall Status: HEALTHY
```

### Failed Validation - Model Not Found

```
======================================================================
PRE-DEPLOYMENT VALIDATION REPORT
======================================================================

✗ ML Models: FAIL
  Status: UNHEALTHY
  Error: /app/models/model1.pkl: Model file not found
  Fix: Train models using: docker compose exec backend python3 scripts/train_ml_models.py. Or specify model paths with --model-paths flag.

✓ Database: PASS
  Status: HEALTHY
  Details: All checks passed

----------------------------------------------------------------------
✗ OVERALL STATUS: NOT READY FOR DEPLOYMENT
  1 of 2 validation(s) failed.
  Please fix the issues above before deploying.
======================================================================
```

### Failed Validation - Database Connection Refused

```
======================================================================
PRE-DEPLOYMENT VALIDATION REPORT
======================================================================

✓ ML Models: PASS
  Status: HEALTHY
  Details: All checks passed

✗ Database: FAIL
  Status: UNHEALTHY
  Error: Error type: connection_refused | Database connection refused. Check if database is running and accessible at the configured host/port.
  Fix: Ensure the database is running: docker compose up -d db. Check DATABASE_URL environment variable is correct.

----------------------------------------------------------------------
✗ OVERALL STATUS: NOT READY FOR DEPLOYMENT
  1 of 2 validation(s) failed.
  Please fix the issues above before deploying.
======================================================================
```

### Failed Validation - Checksum Mismatch

```
======================================================================
PRE-DEPLOYMENT VALIDATION REPORT
======================================================================

✗ ML Models: FAIL
  Status: UNHEALTHY
  Error: [checksum_mismatch] /app/models/model1.pkl: Checksum mismatch. Expected: abc123..., Actual: def456...
  Fix: Model file checksum mismatch. Re-train the model or update the expected checksum in MODEL_CHECKSUMS environment variable or config file.

✓ Database: PASS
  Status: HEALTHY
  Details: All checks passed

----------------------------------------------------------------------
✗ OVERALL STATUS: NOT READY FOR DEPLOYMENT
  1 of 2 validation(s) failed.
  Please fix the issues above before deploying.
======================================================================
```

### Verbose Output Example

```bash
$ docker compose exec backend python3 scripts/validate_deployment.py --verbose

2025-01-25 21:14:32 - __main__ - DEBUG - Verbose mode enabled
2025-01-25 21:14:32 - __main__ - DEBUG - Configuration loaded: {'model_paths': None, 'model_checksums': None, 'database_url': 'postgresql://fpl_user:***@db:5432/fpl_db', 'db_timeout': 2.0, 'model_timeout': None}
2025-01-25 21:14:32 - __main__ - INFO - Starting pre-deployment validation...
2025-01-25 21:14:32 - __main__ - INFO - Starting model validation...
2025-01-25 21:14:32 - app.services.ml.model_file_validator - DEBUG - Validating model: /app/models/attack_model.pkl
2025-01-25 21:14:32 - app.services.ml.model_file_validator - DEBUG - Model file exists: True
2025-01-25 21:14:32 - app.services.ml.model_file_validator - DEBUG - Model file size: 1048576 bytes
2025-01-25 21:14:32 - __main__ - INFO - Model validation: ✓ PASSED
2025-01-25 21:14:32 - __main__ - INFO - Starting database validation...
2025-01-25 21:14:32 - app.services.database_validator - DEBUG - Attempting database connection...
2025-01-25 21:14:32 - app.services.database_validator - DEBUG - Connection established in 0.15s
2025-01-25 21:14:32 - app.services.database_validator - DEBUG - Executing health check query...
2025-01-25 21:14:32 - app.services.database_validator - DEBUG - Query executed successfully in 0.05s
2025-01-25 21:14:32 - __main__ - INFO - Database validation: ✓ PASSED

[Validation report output...]
```

---

## FAQ

### General Questions

**Q: Do I need to run validation manually, or does it happen automatically?**  
A: Both! The validation runs automatically when the API starts (in `main.py` startup event), but you can also run it manually before deployment using `scripts/validate_deployment.py` to catch issues early.

**Q: What happens if validation fails during startup?**  
A: The API startup is aborted, the process exits with code 1, and the container/process stops. This prevents broken deployments from accepting traffic.

**Q: Can I skip validation?**  
A: Not recommended. Validation is a critical safety mechanism. If you need to disable it temporarily for debugging, you would need to modify the code, but this is strongly discouraged.

**Q: How long does validation take?**  
A: Typically 0.2 - 2.5 seconds, usually under 1 second. Model validation is fast (< 0.5s), and database validation usually completes in < 2s with default timeout.

### Configuration Questions

**Q: Do I need to specify MODEL_PATHS?**  
A: No, it's optional. If not specified, the system auto-detects models using PLEngine. However, explicitly specifying paths is recommended for production.

**Q: Should I use checksums in production?**  
A: Yes, highly recommended. Checksums verify model file integrity and detect tampering or corruption. Use checksums in staging and production environments.

**Q: How do I calculate model checksums?**  
A: Use this command:
```bash
docker compose exec backend python3 -c "
import hashlib
with open('/app/models/model1.pkl', 'rb') as f:
    print(hashlib.sha256(f.read()).hexdigest())
"
```

**Q: What timeout values should I use?**  
A: 
- Database timeout: 2.0s (default) for local/fast networks, 5.0s for slow networks
- Model timeout: None (default) for local files, 30.0s for network filesystems
- Performance budget: 5.0s (default), increase to 10.0s if needed

### Troubleshooting Questions

**Q: Validation passes locally but fails in Docker. Why?**  
A: Common causes:
- Database hostname: Use `db` (service name) instead of `localhost` in Docker
- Model file paths: Ensure models are mounted as volumes
- Network: Containers may not be on the same network

**Q: Validation is slow. How can I speed it up?**  
A: 
- Check network latency (ping database host)
- Verify model files are on fast storage (not network-mounted)
- Reduce timeout values if they're too high
- Check database performance

**Q: I get "connection refused" even though the database is running.**  
A: 
- Verify DATABASE_URL uses correct hostname (`db` for Docker, `localhost` for local)
- Check database is listening on the correct port
- Verify network connectivity: `ping db` or `nc -zv db 5432`
- Check firewall rules

**Q: Checksum validation fails after re-training models. What should I do?**  
A: Re-calculate checksums after training:
```bash
# Calculate new checksums
docker compose exec backend python3 -c "
import hashlib
import os
models = ['/app/models/model1.pkl', '/app/models/model2.pkl']
for model in models:
    if os.path.exists(model):
        with open(model, 'rb') as f:
            print(f'{model}:{hashlib.sha256(f.read()).hexdigest()}')
"
# Update MODEL_CHECKSUMS environment variable with new values
```

### CI/CD Questions

**Q: How do I integrate validation into my CI/CD pipeline?**  
A: Add a step that runs `scripts/validate_deployment.py` before deployment. The script returns exit code 0 (success) or 1 (failure), which will automatically fail the pipeline if validation fails. See [CI/CD Integration](#cicd-integration) section for examples.

**Q: Should I validate in every environment (dev, staging, prod)?**  
A: Yes, but with different configurations:
- Development: Basic validation (no checksums, shorter timeouts)
- Staging: Full validation (optional checksums, production-like timeouts)
- Production: Full validation (mandatory checksums, appropriate timeouts)

**Q: How do I handle secrets in CI/CD?**  
A: Use your CI/CD platform's secret management:
- GitHub Actions: Repository secrets
- GitLab CI: CI/CD variables (masked)
- Jenkins: Credentials plugin
- Never commit secrets to version control

### Docker/Kubernetes Questions

**Q: How do I mount model files in Docker?**  
A: Use volumes in `docker-compose.yml`:
```yaml
volumes:
  - ./backend/models:/app/models:ro  # Read-only mount
```

**Q: How do I pass environment variables to containers?**  
A: Multiple ways:
- `.env` file (automatically loaded by docker-compose)
- `environment:` section in docker-compose.yml
- `env_file:` in docker-compose.yml
- Kubernetes: ConfigMaps and Secrets

**Q: Should I use init containers for validation in Kubernetes?**  
A: Yes, init containers are a good pattern. They run before the main container starts and can validate dependencies. See [Docker/Kubernetes Deployment](#dockerkubernetes-deployment) section for examples.

**Q: How do I handle model files in Kubernetes?**  
A: Options:
1. Persistent Volume (recommended for large models)
2. Init container that downloads/trains models
3. ConfigMap (only for very small models, not recommended)

### Performance Questions

**Q: Validation is taking longer than expected. Is this normal?**  
A: Check:
- Network latency to database
- Model file location (local vs network-mounted)
- Database performance
- System load

Typical times: < 1 second total. If consistently > 2 seconds, investigate.

**Q: Should I increase the performance budget?**  
A: Only if:
- You're using network-mounted filesystems (NFS, EFS)
- Database is in a different region/data center
- Model files are very large (> 100MB)

Otherwise, investigate why validation is slow rather than increasing the budget.

### Security Questions

**Q: Are database credentials secure in the validation system?**  
A: Yes, credentials are automatically masked in logs. However, you should still:
- Use environment variables or secrets management
- Never commit credentials to version control
- Use strong passwords
- Rotate credentials regularly

**Q: Should I use checksums in all environments?**  
A: 
- Development: Optional (for convenience)
- Staging: Recommended (to catch issues early)
- Production: Mandatory (for security and integrity)

**Q: How do I securely store model checksums?**  
A: 
- Store in environment variables or secrets management
- Don't commit to version control
- Rotate when models are updated
- Use separate checksums per environment

---

## Additional Resources

- [Main Deployment Guide](../README.md) - Overall deployment documentation
- [Database Population Guide](../backend/DATABASE_POPULATION_GUIDE.md) - Database setup
- [ML Training Guide](../ML_TRAINING_AND_PREDICTION_GUIDE.md) - Model training
- [Environment Variables Guide](../ENVIRONMENT_VARIABLES.md) - Environment configuration

---

## Support

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting Guide](#troubleshooting-guide) section
2. Review validation logs with `--verbose` flag
3. Check application logs: `docker compose logs backend`
4. Verify environment variables: `docker compose exec backend env | grep -E "(DATABASE|MODEL)"`

For additional help, refer to the main project documentation or open an issue in the repository.
