# Docker Disk Space Management

## Issue: No Space Left on Device

When building Docker images, you may encounter:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

This happens when the disk is full (typically >95% usage).

---

## Quick Fix

### 1. Clean Up Docker Resources

```bash
# Remove all unused containers, networks, images, and build cache
docker system prune -a --volumes --force

# Remove only build cache (keeps images)
docker builder prune -a --force

# Remove dangling images
docker image prune -a --force
```

### 2. Check Disk Usage

```bash
# Check overall disk usage
df -h

# Check Docker disk usage
docker system df

# Check largest Docker images
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | sort -k3 -h -r
```

---

## Prevention Strategies

### 1. Regular Cleanup

Add to your workflow:

```bash
# Weekly cleanup script
docker system prune -a --volumes --force
docker builder prune -a --force
```

### 2. Optimize Dockerfiles

**Backend Dockerfile.dev** already includes:
- `--no-cache-dir` for pip
- Cache purging after install
- Removal of Python cache files

**Frontend Dockerfile.dev** uses:
- `npm ci` (cleaner than `npm install`)
- `npm cache clean --force`

### 3. Use Multi-stage Builds

Production Dockerfiles use multi-stage builds to:
- Reduce final image size
- Keep build artifacts separate
- Minimize disk usage

### 4. Limit Build Cache

```bash
# Set build cache size limit
docker system prune --filter "until=24h" --force
```

---

## Disk Space Breakdown

### Typical Usage (38GB disk)

- **System**: ~5-10GB
- **Docker Images**: ~10-15GB (can be cleaned)
- **Docker Volumes**: ~1-2GB (database data)
- **Build Cache**: ~2-5GB (can be cleaned)
- **Application Code**: ~1GB
- **Logs**: ~1-2GB

**Total**: ~20-35GB (leaves 3-18GB free)

### When Disk is Full

1. **Docker images** (largest): 23GB+ can be reclaimed
2. **Build cache**: 2-5GB can be reclaimed
3. **Unused volumes**: 1-2GB can be reclaimed
4. **Logs**: Can be rotated/cleaned

---

## Cleanup Commands

### Full Cleanup (Removes Everything Unused)

```bash
# ⚠️ WARNING: Removes all unused images, containers, volumes, and networks
docker system prune -a --volumes --force
```

### Safe Cleanup (Keeps Running Containers)

```bash
# Remove only stopped containers
docker container prune --force

# Remove unused images (not tagged)
docker image prune --force

# Remove unused volumes (be careful - may remove database data)
docker volume prune --force

# Remove build cache
docker builder prune --force
```

### Selective Cleanup

```bash
# Remove images older than 24 hours
docker image prune -a --filter "until=24h" --force

# Remove build cache older than 1 week
docker builder prune --filter "until=168h" --force

# Remove specific image
docker rmi <image-id>

# Remove specific volume (be careful!)
docker volume rm <volume-name>
```

---

## Monitoring Disk Space

### Check Current Usage

```bash
# Overall disk usage
df -h

# Docker-specific usage
docker system df

# Detailed breakdown
docker system df -v
```

### Set Up Alerts

Add to cron (runs daily):
```bash
# Check disk usage and alert if >90%
df -h / | awk 'NR==2 {if ($5+0 > 90) print "WARNING: Disk usage is " $5}'
```

---

## Build Optimization

### During Build

The Dockerfiles are optimized to minimize disk usage:

**Backend:**
- Uses `--no-cache-dir` for pip
- Cleans pip cache after install
- Removes Python cache files
- Uses multi-stage builds (production)

**Frontend:**
- Uses `npm ci` (cleaner)
- Cleans npm cache
- Uses multi-stage builds (production)

### Build with Cache Management

```bash
# Build without cache (uses more disk temporarily but cleaner)
docker compose build --no-cache

# Build with cache size limit
DOCKER_BUILDKIT=1 docker compose build --progress=plain
```

---

## Emergency Cleanup

If disk is completely full and Docker won't start:

```bash
# 1. Stop all containers
docker compose down

# 2. Remove all unused resources
docker system prune -a --volumes --force

# 3. Remove build cache
docker builder prune -a --force

# 4. Check space
df -h

# 5. If still full, remove specific large images
docker images | grep -E "fpl|python|node" | awk '{print $3}' | xargs docker rmi --force

# 6. Clean logs (if needed)
journalctl --vacuum-time=7d  # Keep only last 7 days
```

---

## Best Practices

1. **Regular Cleanup**: Run cleanup weekly
2. **Monitor Usage**: Check `docker system df` regularly
3. **Use Multi-stage Builds**: Reduces final image size
4. **Remove Unused Images**: Don't keep old versions
5. **Limit Build Cache**: Set retention policies
6. **Clean After Builds**: Remove intermediate layers

---

## Automated Cleanup Script

Create `scripts/docker-cleanup.sh`:

```bash
#!/bin/bash
# Docker cleanup script

echo "Cleaning Docker resources..."

# Remove unused containers
docker container prune --force

# Remove unused images (not tagged)
docker image prune --force

# Remove build cache older than 1 week
docker builder prune --filter "until=168h" --force

# Show current usage
echo "Current Docker disk usage:"
docker system df

echo "Current disk usage:"
df -h /
```

Make it executable:
```bash
chmod +x scripts/docker-cleanup.sh
```

Run weekly:
```bash
# Add to crontab (runs every Sunday at 2 AM)
0 2 * * 0 /path/to/scripts/docker-cleanup.sh
```

---

## Related Documentation

- [Docker Refactoring Summary](./DOCKER_REFACTORING_SUMMARY.md)
- [Docker Troubleshooting](./DOCKER_TROUBLESHOOTING.md)
