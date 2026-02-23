# Terraform Outputs for Cow Lameness Detection Platform

# Networking
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = module.networking.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = module.networking.private_subnet_ids
}

# Load Balancer
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = module.load_balancer.alb_dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = module.load_balancer.alb_zone_id
}

# Database
output "db_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.database.db_endpoint
}

output "db_name" {
  description = "Database name"
  value       = module.database.db_name
}

# Storage
output "efs_file_system_id" {
  description = "EFS file system ID"
  value       = module.storage.efs_file_system_id
}

output "s3_bucket_name" {
  description = "S3 bucket name for backups"
  value       = module.storage.s3_bucket_name
}

# ECS
output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = module.ecs.cluster_name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = module.ecs.cluster_arn
}

# GPU Worker
output "gpu_worker_asg_name" {
  description = "Name of the GPU worker Auto Scaling Group"
  value       = module.gpu_worker.asg_name
}

output "gpu_enabled" {
  description = "Whether GPU worker is currently enabled"
  value       = var.gpu_enabled
}

# Application URL
output "application_url" {
  description = "URL to access the application"
  value       = "https://${module.load_balancer.alb_dns_name}"
}

# Service Discovery
output "service_discovery_namespace" {
  description = "Service discovery namespace for internal DNS"
  value       = module.ecs.service_discovery_namespace
}

# Video Storage
output "videos_bucket_name" {
  description = "S3 bucket name for videos"
  value       = module.storage.videos_bucket_name
}

output "cloudfront_domain_name" {
  description = "CloudFront domain for video streaming"
  value       = module.storage.cloudfront_domain_name
}
