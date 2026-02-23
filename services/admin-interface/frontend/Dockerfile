# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY services/admin-interface/frontend/package*.json ./
RUN npm install

# Copy source code
COPY services/admin-interface/frontend/ ./

# Build
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY services/admin-interface/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]

