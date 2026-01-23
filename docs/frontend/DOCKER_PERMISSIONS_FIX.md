# Docker Permissions Fix for Frontend

## Issue

When running `npm install` inside the frontend container, you encountered:
```
npm error code EACCES
npm error syscall open
npm error path /app/package-lock.json
npm error errno -13
npm error [Error: EACCES: permission denied, open '/app/package-lock.json']
```

## Root Cause

The production Dockerfile (`Dockerfile`) runs as the `nextjs` user (uid 1001) for security, but:
1. Files on the host are owned by root
2. No volume mounts were configured for development
3. The container couldn't write to `/app/package-lock.json` due to permission mismatch

## Solution

### 1. Created Development Dockerfile

**File**: `frontend/Dockerfile.dev`

A simplified Dockerfile for development that:
- Runs as root (for development convenience)
- Installs all dependencies
- Supports hot-reload with volume mounts

### 2. Updated docker-compose.yml

Added development configuration:
- Uses `Dockerfile.dev` instead of production `Dockerfile`
- Mounts `./frontend:/app` for hot-reload
- Uses anonymous volumes for `node_modules` and `.next` to prevent overwriting
- Sets `user: root` for development

### Changes Made

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile.dev  # Use dev Dockerfile
  command: npm run dev
  volumes:
    - ./frontend:/app           # Mount source for hot-reload
    - /app/node_modules         # Preserve node_modules
    - /app/.next                # Preserve build cache
  user: root                     # Run as root for dev
```

## Usage

### Install Dependencies

```bash
docker compose exec frontend npm install
```

### Run Development Server

```bash
docker compose up frontend
```

Or in detached mode:
```bash
docker compose up -d frontend
```

### Rebuild After Dockerfile Changes

```bash
docker compose build frontend
docker compose up -d frontend
```

## Production vs Development

### Production (`Dockerfile`)
- Multi-stage build
- Runs as `nextjs` user (uid 1001) for security
- Optimized for size and security
- No volume mounts
- Uses `npm run build` and serves static files

### Development (`Dockerfile.dev`)
- Single-stage build
- Runs as root (for convenience)
- Volume mounts for hot-reload
- Uses `npm run dev` for development server

## Security Note

⚠️ **Important**: The development setup runs as root for convenience. This is acceptable for local development but should **never** be used in production.

For production deployments:
- Use the original `Dockerfile` (production)
- Remove `user: root` from docker-compose.yml
- Don't mount volumes in production

## Troubleshooting

### Still Getting Permission Errors?

1. **Check file ownership**:
   ```bash
   ls -la frontend/
   ```

2. **Fix permissions** (if needed):
   ```bash
   chmod -R 755 frontend/
   ```

3. **Rebuild container**:
   ```bash
   docker compose build --no-cache frontend
   docker compose up -d frontend
   ```

### node_modules Not Updating?

The anonymous volume `/app/node_modules` preserves node_modules. To reset:
```bash
docker compose down frontend
docker volume prune  # Be careful - removes unused volumes
docker compose up -d frontend
```

### Changes Not Reflecting?

1. Ensure volume mount is working:
   ```bash
   docker compose exec frontend ls -la /app
   ```

2. Check if files are synced:
   ```bash
   docker compose exec frontend cat /app/package.json
   ```

## Related Files

- `frontend/Dockerfile` - Production Dockerfile
- `frontend/Dockerfile.dev` - Development Dockerfile
- `docker-compose.yml` - Docker Compose configuration
