# Docker Troubleshooting Guide

## Common Issues and Solutions

### 1. Database Authentication Errors

#### Symptom
```
FATAL: password authentication failed for user "postgres"
DETAIL: Role "postgres" does not exist.
```

#### Cause
External tools, monitoring scripts, or health checks are trying to connect using the default "postgres" user, which doesn't exist in this setup.

#### Solution
**These errors are harmless** and don't affect functionality. The application uses the correct user (`fpl_db_user` from `.env`).

To verify your services are working:
```bash
# Check service status
docker compose ps

# Test backend health
curl http://localhost:8000/health

# Test database connection
docker compose exec db psql -U fpl_db_user -d fpl_db -c "SELECT 1;"
```

#### Suppressing Logs (Optional)
If the errors are too noisy, you can suppress them by adjusting PostgreSQL log levels in `docker-compose.yml`:

```yaml
environment:
  POSTGRES_INITDB_ARGS: "-E UTF8 --locale=C"
```

Or filter logs:
```bash
docker compose logs db 2>&1 | grep -v "password authentication failed"
```

---

### 2. Container Won't Start

#### Symptom
Container exits immediately or shows "unhealthy" status.

#### Solutions

**Check logs:**
```bash
docker compose logs <service-name>
```

**Check resource limits:**
```bash
docker stats
```

**Verify environment variables:**
```bash
docker compose config
```

**Rebuild containers:**
```bash
docker compose down
docker compose build --no-cache
docker compose up
```

---

### 3. Database Connection Issues

#### Symptom
Backend can't connect to database.

#### Solutions

**Verify database is healthy:**
```bash
docker compose ps db
# Should show "healthy" status
```

**Check database credentials:**
```bash
# Verify .env file has correct values
cat .env | grep POSTGRES

# Test connection manually
docker compose exec db psql -U fpl_db_user -d fpl_db
```

**Check DATABASE_URL format:**
```bash
# Should be: postgresql://user:password@db:5432/database
echo $DATABASE_URL
```

**Restart services:**
```bash
docker compose restart backend db
```

---

### 4. Permission Denied Errors

#### Symptom
```
EACCES: permission denied, open '/app/package-lock.json'
```

#### Solution
This happens when containers run as non-root but files are owned by root.

**For development:**
- Use `Dockerfile.dev` which runs as root
- Or fix permissions: `chmod -R 755 frontend/`

**For production:**
- Use production Dockerfiles (non-root users)
- Ensure files have correct ownership

See [Docker Permissions Fix](../frontend/DOCKER_PERMISSIONS_FIX.md) for details.

---

### 5. Port Already in Use

#### Symptom
```
Error: bind: address already in use
```

#### Solution

**Find process using the port:**
```bash
# For port 8000 (backend)
lsof -i :8000
# or
netstat -tulpn | grep 8000

# For port 3000 (frontend)
lsof -i :3000

# For port 5432 (database)
lsof -i :5432
```

**Kill the process or change port:**
```yaml
# In docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

---

### 6. Out of Memory Errors

#### Symptom
```
Container killed (OOM)
```

#### Solution

**Check current usage:**
```bash
docker stats
```

**Adjust memory limits in docker-compose.yml:**
```yaml
mem_limit: 1536m  # Increase if needed
```

**Remember:** Total system has 4GB RAM, so:
- Database: 512MB
- Backend: 1536MB
- Frontend: 1GB
- **Total: ~3GB** (leaves 1GB for system)

---

### 7. Health Check Failures

#### Symptom
Container shows "unhealthy" status.

#### Solutions

**Check health check endpoint:**
```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Database
docker compose exec db pg_isready -U fpl_db_user
```

**Adjust health check timing:**
```yaml
healthcheck:
  start_period: 60s  # Increase if service needs more time
  interval: 30s
  timeout: 10s
  retries: 3
```

---

### 8. Volume Mount Issues

#### Symptom
Changes not reflecting in container.

#### Solutions

**Verify volume mounts:**
```bash
docker compose exec backend ls -la /app
docker compose exec frontend ls -la /app
```

**Restart with clean volumes:**
```bash
docker compose down -v
docker compose up
```

**Check .dockerignore:**
- Ensure important files aren't being ignored

---

### 9. Build Failures

#### Symptom
```
ERROR: failed to solve: process "/bin/sh -c ..." did not complete successfully
```

#### Solutions

**Clear build cache:**
```bash
docker compose build --no-cache
```

**Check Dockerfile syntax:**
```bash
docker build -t test -f backend/Dockerfile.dev backend/
```

**Verify dependencies:**
- Check `requirements.txt` (backend)
- Check `package.json` (frontend)

---

### 10. Network Issues

#### Symptom
Services can't communicate with each other.

#### Solutions

**Verify network exists:**
```bash
docker network ls | grep fpl
```

**Check service connectivity:**
```bash
# From backend container
docker compose exec backend curl http://db:5432

# From frontend container
docker compose exec frontend curl http://backend:8000/health
```

**Recreate network:**
```bash
docker compose down
docker compose up
```

---

## Quick Diagnostic Commands

```bash
# Check all service status
docker compose ps

# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f backend

# Check resource usage
docker stats

# Verify configuration
docker compose config

# Test database connection
docker compose exec db psql -U fpl_db_user -d fpl_db -c "SELECT version();"

# Test backend API
curl http://localhost:8000/health

# Test frontend
curl http://localhost:3000

# Restart all services
docker compose restart

# Stop and remove everything
docker compose down -v
```

---

## Getting Help

1. **Check logs first**: `docker compose logs <service>`
2. **Verify configuration**: `docker compose config`
3. **Check resource usage**: `docker stats`
4. **Review documentation**: See [Docker Refactoring Summary](./DOCKER_REFACTORING_SUMMARY.md)

---

## Related Documentation

- [Docker Refactoring Summary](./DOCKER_REFACTORING_SUMMARY.md)
- [Docker Permissions Fix](../frontend/DOCKER_PERMISSIONS_FIX.md)
- [Environment Variables](../ENVIRONMENT_VARIABLES.md)
