#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/user-data.log) 2>&1

echo "Starting GPU worker setup..."

# Install required packages
yum update -y
yum install -y amazon-efs-utils docker jq

# Start Docker
systemctl enable docker
systemctl start docker

# Install docker-compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create data directory and mount EFS
mkdir -p /app/data
mount -t efs -o tls ${efs_file_system_id}:/ /app/data
echo "${efs_file_system_id}:/ /app/data efs _netdev,tls 0 0" >> /etc/fstab

# Create directories for GPU services
mkdir -p /app/services
cd /app/services

# Login to ECR if registry is provided
if [ -n "${ecr_registry}" ]; then
    aws ecr get-login-password --region $(curl -s http://169.254.169.254/latest/meta-data/placement/region) | docker login --username AWS --password-stdin ${ecr_registry}
fi

# Create docker-compose file for GPU services
cat > /app/services/docker-compose.yml << 'EOF'
version: '3.8'

services:
%{ for service in split(" ", gpu_services) ~}
  ${service}:
    image: ${ecr_registry != "" ? "${ecr_registry}/${service}:latest" : "${service}:latest"}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NATS_URL=nats://${nats_endpoint}
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - /app/data:/app/data
    restart: unless-stopped
    logging:
      driver: awslogs
      options:
        awslogs-group: /ec2/${name_prefix}-gpu-worker
        awslogs-region: REGION_PLACEHOLDER
        awslogs-stream-prefix: ${service}

%{ endfor ~}
EOF

# Replace region placeholder
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
sed -i "s/REGION_PLACEHOLDER/$REGION/g" /app/services/docker-compose.yml

# Pull images
echo "Pulling GPU service images..."
docker-compose pull || true

# Start GPU services
echo "Starting GPU services..."
docker-compose up -d

echo "GPU worker setup complete!"
