# Build stage
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application.
# VITE_API_URL is intentionally NOT set: the client falls back to same-origin
# ("/api/v1/...") so requests go through the nginx reverse proxy in this image.
# Override with --build-arg VITE_API_URL=https://api.example.com only when the
# frontend and backend live on different origins and there's no proxy in front.
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build


# Runtime stage
FROM nginx:alpine

# Copy nginx config template; nginx:alpine entrypoint runs envsubst on files
# in /etc/nginx/templates at container start, so BACKEND_HOST / BACKEND_PORT
# can be overridden per environment (docker-compose service name, host gateway,
# external IP, etc.) without rebuilding the image.
COPY --from=builder /app/nginx.conf.template /etc/nginx/templates/default.conf.template

# Defaults match the classic docker-compose wiring (service name "backend" on
# port 8000). Override with `-e BACKEND_HOST=... -e BACKEND_PORT=...`.
ENV BACKEND_HOST=backend
ENV BACKEND_PORT=8000

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]
