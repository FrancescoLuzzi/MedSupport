# Multi-stage build: compile frontend, then serve with Caddy + TLS

# Stage 1: build the frontend assets
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json .
RUN npm install
COPY ./frontend .
# Build static assets into dist/frontend
RUN npm run build

# Stage 2: production image using Caddy
FROM caddy:2-alpine

# Copy Caddy configuration and TLS certificates
COPY Caddyfile /etc/caddy/Caddyfile
COPY cert.pem key.pem /etc/caddy/

# Serve built frontend and static backend assets
COPY --from=builder /app/dist/frontend /srv
COPY ./db /srv/db
COPY ./models/paraphrase-italian-mpnet-med-v2/final /srv/models/paraphrase-italian-mpnet-med-v2/final

EXPOSE 443
CMD ["caddy", "run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
