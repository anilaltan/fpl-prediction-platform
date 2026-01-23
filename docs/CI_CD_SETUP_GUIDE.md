# CI/CD Pipeline Setup Guide

This guide provides step-by-step instructions for setting up the GitHub Actions CI/CD pipeline for the FPL Prediction Platform.

## Prerequisites

- GitHub repository with Actions enabled
- Ubuntu VPS (2 vCPU, 4GB RAM) with SSH access
- Docker and Docker Compose installed on VPS
- GitHub account with repository access

---

## Phase 1: Repository Configuration

### Step 1: Update docker-compose.prod.yml

**Location**: `/root/fpl-prediction-platform/docker-compose.prod.yml`

**Action**: Replace the placeholder repository path with your actual GitHub repository.

```yaml
# Find these lines:
image: ${GHCR_BACKEND_IMAGE:-ghcr.io/your-org/fpl-prediction-platform/backend:latest}
image: ${GHCR_FRONTEND_IMAGE:-ghcr.io/your-org/fpl-prediction-platform/frontend:latest}

# Replace 'your-org/fpl-prediction-platform' with your actual:
# - GitHub username or organization name
# - Repository name
# Example: ghcr.io/johndoe/fpl-prediction-platform/backend:latest
```

**Alternative**: Set environment variables on VPS instead of hardcoding:
```bash
export GHCR_BACKEND_IMAGE=ghcr.io/YOUR_USERNAME/YOUR_REPO/backend:latest
export GHCR_FRONTEND_IMAGE=ghcr.io/YOUR_USERNAME/YOUR_REPO/frontend:latest
```

---

## Phase 2: SSH Key Setup

### Step 2: Generate SSH Key Pair

**On your local machine**:

```bash
# Generate new SSH key (or use existing)
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/vps_deploy_key

# This creates:
# - ~/.ssh/vps_deploy_key (private key - keep secret!)
# - ~/.ssh/vps_deploy_key.pub (public key)
```

**Important**: 
- Do NOT add a passphrase (GitHub Actions can't handle interactive prompts)
- Keep the private key secure - you'll add it to GitHub Secrets

### Step 3: Add Public Key to VPS

**Option A: Using ssh-copy-id** (if you have existing SSH access):
```bash
ssh-copy-id -i ~/.ssh/vps_deploy_key.pub user@your-vps-ip
```

**Option B: Manual method**:
```bash
# 1. Display public key
cat ~/.ssh/vps_deploy_key.pub

# 2. SSH into VPS
ssh user@your-vps-ip

# 3. On VPS, add to authorized_keys
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Option C: Add to root user** (if deploying as root):
```bash
# On VPS as root
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo "PASTE_PUBLIC_KEY_HERE" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
```

### Step 4: Test SSH Connection

```bash
# Test connection from local machine
ssh -i ~/.ssh/vps_deploy_key user@your-vps-ip

# Should connect without password prompt
```

---

## Phase 3: GitHub Configuration

### Step 5: Create GitHub Personal Access Token (PAT)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: `GHCR-Deploy-Token`
4. Select scopes:
   - ✅ `read:packages` (to pull images)
   - ✅ `write:packages` (to push images, if needed)
5. Click "Generate token"
6. **Copy the token immediately** - you won't see it again!

### Step 6: Configure GitHub Repository Secrets

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** for each:

#### Required Secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `VPS_HOST` | `your-vps-ip-or-domain.com` | VPS IP address or hostname |
| `VPS_USER` | `root` or `ubuntu` | SSH username for VPS |
| `VPS_SSH_KEY` | Content of `~/.ssh/vps_deploy_key` | Private SSH key (entire content) |
| `VPS_PORT` | `22` (optional) | SSH port (default: 22) |
| `VPS_DEPLOY_PATH` | `/opt/fpl-prediction-platform` (optional) | Deployment directory on VPS |
| `GHCR_TOKEN` | Your PAT from Step 5 | GitHub Personal Access Token for GHCR |

**To get private key content**:
```bash
cat ~/.ssh/vps_deploy_key
# Copy entire output including -----BEGIN and -----END lines
```

**Important**: 
- `VPS_SSH_KEY` should include the entire private key file content
- `GHCR_TOKEN` is your Personal Access Token from Step 5

---

## Phase 4: VPS Preparation

### Step 7: Install Docker and Docker Compose on VPS

**SSH into VPS** and run:

```bash
# Update system
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (if not root)
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

# Install Docker Compose (plugin)
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### Step 8: Create Deployment Directory

```bash
# Create directory
sudo mkdir -p /opt/fpl-prediction-platform
sudo chown $USER:$USER /opt/fpl-prediction-platform

# Or if deploying as root:
mkdir -p /opt/fpl-prediction-platform
```

### Step 9: Clone Repository or Copy Files

**Option A: Clone repository** (recommended for updates):
```bash
cd /opt/fpl-prediction-platform
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .
# Or use SSH: git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git .
```

**Option B: Copy files manually**:
```bash
# From local machine
scp docker-compose.prod.yml user@vps:/opt/fpl-prediction-platform/
scp -r backend/scripts/init_timescaledb.sql user@vps:/opt/fpl-prediction-platform/backend/scripts/
```

### Step 10: Create Production .env File

```bash
# On VPS
cd /opt/fpl-prediction-platform
nano .env
```

**Required environment variables**:
```bash
# Database
POSTGRES_USER=fpl_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=fpl_db
DATABASE_URL=postgresql://fpl_user:your_secure_password_here@db:5432/fpl_db

# Application
SECRET_KEY=your_secret_key_here_generate_with_openssl_rand_hex_32
FPL_EMAIL=your_fpl_email@example.com
FPL_PASSWORD=your_fpl_password_here

# Frontend
NEXT_PUBLIC_API_URL=http://your-vps-ip:8000
# Or if using domain: NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

**Generate SECRET_KEY**:
```bash
openssl rand -hex 32
```

### Step 11: Configure GHCR Authentication on VPS

**Option A: Using PAT** (recommended):
```bash
# On VPS
echo $GHCR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

**Option B: Using Docker config file**:
```bash
# On VPS
mkdir -p ~/.docker
cat > ~/.docker/config.json << EOF
{
  "auths": {
    "ghcr.io": {
      "auth": "$(echo -n YOUR_GITHUB_USERNAME:YOUR_PAT | base64)"
    }
  }
}
EOF
chmod 600 ~/.docker/config.json
```

**Option C: For root user**:
```bash
# Same as above but use /root/.docker/config.json
```

### Step 12: Test Image Pull

```bash
# On VPS, test pulling an image (after first build)
docker pull ghcr.io/YOUR_USERNAME/YOUR_REPO/backend:latest

# Should succeed without authentication errors
```

---

## Phase 5: Testing CI/CD

### Step 13: Test CI Workflow Locally

**Backend**:
```bash
cd backend

# Install dependencies
pip install ruff mypy
pip install -r requirements.txt

# Run checks
ruff check .
ruff format --check .
mypy app/ --ignore-missing-imports --no-strict-optional
```

**Frontend**:
```bash
cd frontend

# Install dependencies
npm ci --legacy-peer-deps

# Run checks
npm run lint
npm run build
```

### Step 14: Create Test PR

1. Create a new branch:
   ```bash
   git checkout -b test-ci-pipeline
   ```

2. Make a small change (e.g., update README)

3. Commit and push:
   ```bash
   git add .
   git commit -m "Test CI pipeline"
   git push origin test-ci-pipeline
   ```

4. Open PR on GitHub

5. Check **Actions** tab - CI workflow should run automatically

### Step 15: Fix CI Issues

- **Ruff errors**: Fix linting issues or add `# noqa` comments
- **MyPy errors**: Fix type issues or add `# type: ignore` comments
- **ESLint errors**: Fix frontend linting issues
- **Build errors**: Fix TypeScript or build configuration issues

### Step 16: Trigger CD Workflow

**Option A: Merge to main**:
```bash
# After CI passes, merge PR to main branch
# This automatically triggers CD workflow
```

**Option B: Manual trigger**:
1. Go to **Actions** tab
2. Select **"CD - Build and Deploy"** workflow
3. Click **"Run workflow"**
4. Select branch (usually `main`)
5. Click **"Run workflow"**

### Step 17: Monitor Deployment

1. Go to **Actions** tab
2. Click on the running workflow
3. Watch for:
   - ✅ `build-and-push` job completes
   - ✅ Images pushed to GHCR (check Packages tab)
   - ✅ `deploy` job connects to VPS
   - ✅ Deployment script executes successfully

### Step 18: Verify Deployment on VPS

**SSH into VPS**:
```bash
cd /opt/fpl-prediction-platform

# Check running containers
docker compose -f docker-compose.prod.yml ps

# Check logs
docker compose -f docker-compose.prod.yml logs backend
docker compose -f docker-compose.prod.yml logs frontend

# Check container health
docker compose -f docker-compose.prod.yml ps
# Should show all containers as "Up (healthy)" or "Up"
```

### Step 19: Test Production Endpoints

```bash
# Test backend health
curl http://your-vps-ip:8000/health

# Test frontend
curl http://your-vps-ip:3000

# Or open in browser:
# - Backend: http://your-vps-ip:8000
# - Frontend: http://your-vps-ip:3000
```

---

## Phase 6: Production Hardening (Optional)

### Step 20: Configure Firewall

```bash
# On VPS, allow required ports
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # Backend
sudo ufw allow 3000/tcp  # Frontend
sudo ufw enable
```

### Step 21: Set Up Reverse Proxy (Nginx)

**Install Nginx**:
```bash
sudo apt-get install nginx
```

**Configure Nginx** (example):
```nginx
# /etc/nginx/sites-available/fpl-platform
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Enable site**:
```bash
sudo ln -s /etc/nginx/sites-available/fpl-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Step 22: Set Up SSL (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal is set up automatically
```

### Step 23: Configure Monitoring (Optional)

- Set up health check monitoring
- Configure log aggregation
- Set up alerts for container failures
- Monitor resource usage (CPU, memory)

---

## Troubleshooting

### CI Workflow Fails

**Ruff/MyPy errors**:
- Fix linting issues in code
- Or temporarily disable strict checks

**Frontend build fails**:
- Check `next.config.js` configuration
- Verify all dependencies are in `package.json`

### CD Workflow Fails

**SSH connection fails**:
- Verify `VPS_SSH_KEY` secret is correct (includes full private key)
- Test SSH connection manually: `ssh -i ~/.ssh/vps_deploy_key user@vps`
- Check VPS firewall allows SSH

**Docker login fails**:
- Verify `GHCR_TOKEN` secret is correct
- Check PAT has `read:packages` scope
- Test login manually on VPS

**Image pull fails**:
- Verify image names in `docker-compose.prod.yml` match repository
- Check GHCR authentication on VPS
- Ensure images were built and pushed successfully

### Deployment Issues

**Containers won't start**:
- Check logs: `docker compose -f docker-compose.prod.yml logs`
- Verify `.env` file exists and has correct values
- Check database initialization script path

**Out of memory**:
- Reduce `mem_limit` values in `docker-compose.prod.yml`
- Check system memory: `free -h`
- Stop unnecessary services

**Port conflicts**:
- Check if ports are in use: `sudo netstat -tulpn | grep :8000`
- Change ports in `docker-compose.prod.yml` if needed

---

## Maintenance

### Updating Deployment

1. Push changes to `main` branch
2. CD workflow automatically:
   - Builds new images
   - Pushes to GHCR
   - Deploys to VPS

### Manual Deployment

```bash
# SSH into VPS
ssh user@vps

# Navigate to deployment directory
cd /opt/fpl-prediction-platform

# Pull latest images
docker compose -f docker-compose.prod.yml pull

# Restart services
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps
```

### Rollback

```bash
# On VPS, pull previous image tag
docker pull ghcr.io/YOUR_REPO/backend:previous-tag

# Update docker-compose.prod.yml to use specific tag
# Then restart
docker compose -f docker-compose.prod.yml up -d
```

---

## Security Best Practices

1. ✅ Use strong passwords for database and secrets
2. ✅ Rotate `SECRET_KEY` regularly
3. ✅ Use SSH keys instead of passwords
4. ✅ Keep Docker and system packages updated
5. ✅ Use reverse proxy with SSL for production
6. ✅ Restrict firewall to necessary ports only
7. ✅ Regularly review GitHub Actions logs
8. ✅ Use least-privilege SSH user (not root if possible)
9. ✅ Store secrets in GitHub Secrets, never in code
10. ✅ Enable 2FA on GitHub account

---

## Support

For issues or questions:
- Check GitHub Actions logs for detailed error messages
- Review Docker logs: `docker compose -f docker-compose.prod.yml logs`
- Verify all secrets are correctly configured
- Test each component individually (SSH, Docker, GHCR)
