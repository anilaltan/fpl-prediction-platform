FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm install --production=false && npm cache clean --force

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Ensure public directory exists (Next.js may need it)
# Create .gitkeep if directory is empty so COPY won't fail
RUN mkdir -p public && touch public/.gitkeep
RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files from builder
# Copy public directory (will have at least .gitkeep, or actual files if added)
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
# Remove .gitkeep if it exists and directory has other files, otherwise keep it
RUN if [ -f ./public/.gitkeep ] && [ "$(find ./public -mindepth 1 -maxdepth 1 ! -name '.gitkeep' | wc -l)" -gt 0 ]; then \
        rm ./public/.gitkeep; \
    fi || true
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
