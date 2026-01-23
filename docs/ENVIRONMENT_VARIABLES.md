# Environment Variables Reference

Complete reference for all environment variables used in the FPL Prediction Platform.

## Quick Start

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your actual values

3. **Never commit `.env` to version control** (it's in `.gitignore`)

---

## Required Variables

### Database Configuration

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `POSTGRES_USER` | PostgreSQL username | `fpl_user` | ✅ Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | `secure_password_123` | ✅ Yes |
| `POSTGRES_DB` | Database name | `fpl_db` | ✅ Yes |
| `DATABASE_URL` | Full connection string | `postgresql://user:pass@host:port/db` | ✅ Yes |

**DATABASE_URL Format**:
- **Docker**: `postgresql://fpl_user:password@db:5432/fpl_db` (use service name `db`)
- **Local**: `postgresql://fpl_user:password@localhost:5432/fpl_db`
- **Note**: ETLService automatically converts to async format (`postgresql+asyncpg://`)

### Application Security

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | Application secret key | `openssl rand -hex 32` | ✅ Yes |

**Generate Secret Key**:
```bash
openssl rand -hex 32
```

---

## Optional Variables

### FPL API Credentials

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `FPL_EMAIL` | FPL account email | `user@example.com` | ❌ No |
| `FPL_PASSWORD` | FPL account password | `password123` | ❌ No |

**Note**: FPL API authentication is optional. Most endpoints work without credentials. Only set these if you need authenticated access to private endpoints.

### Frontend Configuration

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `NEXT_PUBLIC_API_URL` | Backend URL for client-side requests | `http://localhost:8000` | ✅ Yes (Frontend) |
| `BACKEND_URL` | Backend URL for server-side API routes | `http://backend:8000` | ✅ Yes (Frontend) |

**NEXT_PUBLIC_API_URL**:
- Used by browser/client-side code
- `NEXT_PUBLIC_` prefix makes it available in browser (Next.js convention)
- Development: `http://localhost:8000`
- Production: Your production backend URL

**BACKEND_URL**:
- Used by Next.js API routes (`app/api/*`) to proxy requests
- Docker: `http://backend:8000` (service name in docker-compose)
- Local: `http://localhost:8000`

---

## Docker Environment Variables

These are automatically set by `docker-compose.yml`. You typically don't need to set them manually:

| Variable | Value | Purpose |
|----------|-------|---------|
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Reduce TensorFlow logging noise |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Disable TensorFlow optimizations |
| `PYTHONWARNINGS` | `ignore::UserWarning` | Suppress Python warnings |

---

## Environment-Specific Configurations

### Development (Local)

```bash
# Database
DATABASE_URL=postgresql://fpl_user:password@localhost:5432/fpl_db

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
```

### Docker (docker-compose)

```bash
# Database (use service name 'db')
DATABASE_URL=postgresql://fpl_user:password@db:5432/fpl_db

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000  # Browser access
BACKEND_URL=http://backend:8000            # Server-side API routes
```

### Production

```bash
# Use production database
DATABASE_URL=postgresql://user:pass@prod-db-host:5432/fpl_db

# Use production backend URL
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
BACKEND_URL=https://api.yourdomain.com

# Strong secret key
SECRET_KEY=<generate_with_openssl_rand_hex_32>
```

---

## Security Best Practices

1. **Never commit `.env` to version control**
   - `.env` is in `.gitignore`
   - Use `.env.example` as template

2. **Use strong passwords**
   - Database password: At least 16 characters, mix of letters, numbers, symbols
   - Secret key: Generate with `openssl rand -hex 32`

3. **Rotate credentials regularly**
   - Change passwords periodically
   - Regenerate SECRET_KEY when compromised

4. **Environment-specific files**
   - `.env.local` - Local overrides (gitignored)
   - `.env.production` - Production values (gitignored)
   - `.env.development` - Development values (gitignored)

5. **Use secrets management in production**
   - Consider using AWS Secrets Manager, HashiCorp Vault, or similar
   - Never hardcode credentials in code

---

## Variable Usage in Codebase

### Backend

**Database**:
- `app/database.py` - Uses `DATABASE_URL` for SQLAlchemy engine
- `app/services/etl_service.py` - Uses `DATABASE_URL` for async engine

**FPL API**:
- `app/services/fpl/service.py` - Uses `FPL_EMAIL` and `FPL_PASSWORD` (optional)

**Security**:
- `SECRET_KEY` - Reserved for future use (JWT, sessions, etc.)

### Frontend

**API Routes** (`app/api/*/route.ts`):
- Uses `BACKEND_URL` for server-side requests to backend

**Client Components**:
- Uses `NEXT_PUBLIC_API_URL` for browser requests (if needed)

---

## Troubleshooting

### Database Connection Issues

**Error**: `Connection refused` or `database does not exist`

**Solutions**:
1. Check `DATABASE_URL` format is correct
2. For Docker: Use service name `db` not `localhost`
3. Verify database container is running: `docker compose ps`
4. Check database credentials match docker-compose.yml

### Frontend API Errors

**Error**: `Failed to fetch` or `Network error`

**Solutions**:
1. Verify `BACKEND_URL` is correct for your environment
2. For Docker: Use `http://backend:8000` (service name)
3. For local: Use `http://localhost:8000`
4. Check backend is running: `docker compose ps backend`

### Missing Environment Variables

**Error**: `None` or `undefined` values

**Solutions**:
1. Ensure `.env` file exists in project root
2. Verify variable names match exactly (case-sensitive)
3. Restart Docker containers after changing `.env`: `docker compose restart`
4. For Next.js: Restart dev server after changing `NEXT_PUBLIC_*` variables

---

## Related Documentation

- [Database Setup Guide](backend/DATABASE_POPULATION_GUIDE.md)
- [Docker Compose Configuration](../docker-compose.yml)
- [Architecture Map](../ARCHITECTURE_MAP.md)
