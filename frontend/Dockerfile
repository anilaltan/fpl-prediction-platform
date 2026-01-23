# =============================================================================
# Frontend Production Dockerfile - Multi-stage build for Next.js
# Optimized for minimal image size and security
# =============================================================================

FROM node:18-alpine AS base

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Metadata labels
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="FPL Prediction Platform" \
      org.opencontainers.image.url="https://github.com/your-org/fpl-prediction-platform" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="FPL Prediction Platform Frontend" \
      org.opencontainers.image.description="Next.js frontend for FPL point prediction"

# Install curl for health checks
RUN apk add --no-cache curl

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS deps

WORKDIR /app

# Copy package files first for better layer caching
COPY package*.json ./

# Install dependencies
# --legacy-peer-deps: Resolves peer dependency conflicts
# --production=false: Install all dependencies (needed for build)
RUN npm ci --legacy-peer-deps && npm cache clean --force

# =============================================================================
# Builder Stage
# =============================================================================
FROM base AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application code
COPY . .

# Ensure public directory exists (Next.js requirement)
RUN mkdir -p public && touch public/.gitkeep

# Build Next.js application
# Output: standalone mode for minimal production image
RUN npm run build

# Remove .gitkeep if public directory has actual files
RUN if [ -f ./public/.gitkeep ] && [ "$(find ./public -mindepth 1 -maxdepth 1 ! -name '.gitkeep' | wc -l)" -gt 0 ]; then \
        rm ./public/.gitkeep; \
    fi || true

# =============================================================================
# Production Runner Stage
# =============================================================================
FROM base AS runner

WORKDIR /app

# Production environment variables
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy necessary files from builder
# Copy public directory
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

# Copy standalone build output (minimal production bundle)
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./

# Copy static assets
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Run the application
CMD ["node", "server.js"]
