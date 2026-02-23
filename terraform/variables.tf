# Terraform Variables for Cow Lameness Detection Platform

# General
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "cow-lameness"
}

variable "environment" {
  description = "Environment (production, staging, development)"
  type        = string
  default     = "production"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Database
variable "db_password" {
  description = "Password for RDS PostgreSQL database"
  type        = string
  sensitive   = true
}

# Security
variable "jwt_secret" {
  description = "JWT secret key for authentication"
  type        = string
  sensitive   = true
}

# Container Registry
variable "ecr_registry" {
  description = "ECR registry URL (e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com)"
  type        = string
  default     = ""
}

# SSL Certificate
variable "certificate_arn" {
  description = "ARN of ACM certificate for HTTPS (optional)"
  type        = string
  default     = ""
}

# GPU Worker Configuration
variable "gpu_enabled" {
  description = "Enable GPU worker instance for ML processing"
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU worker"
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot_instances" {
  description = "Use spot instances for GPU worker (70% cost savings)"
  type        = bool
  default     = true
}

# CloudFront Configuration
variable "enable_cloudfront" {
  description = "Enable CloudFront CDN for video streaming (requires cloudfront:* IAM permissions)"
  type        = bool
  default     = false
}

# Container Image Tag
variable "image_tag" {
  description = "Docker image tag to use for custom services (e.g., latest, v20260106-amd64)"
  type        = string
  default     = "v20260106-amd64"
}
