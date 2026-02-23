# GPU Worker Module Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_subnet_ids" {
  description = "IDs of private subnets"
  type        = list(string)
}

variable "gpu_security_group_id" {
  description = "Security group ID for GPU worker"
  type        = string
}

variable "gpu_enabled" {
  description = "Whether GPU worker should be running"
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU worker"
  type        = string
  default     = "g4dn.xlarge"
}

variable "use_spot_instances" {
  description = "Use spot instances for GPU worker"
  type        = bool
  default     = true
}

variable "efs_file_system_id" {
  description = "EFS file system ID to mount"
  type        = string
}

variable "ecr_registry" {
  description = "ECR registry URL"
  type        = string
  default     = ""
}

variable "nats_endpoint" {
  description = "NATS endpoint for GPU services"
  type        = string
}

variable "gpu_services" {
  description = "List of GPU services to run"
  type        = list(string)
  default = [
    "yolo-pipeline",
    "sam3-pipeline",
    "dinov3-pipeline",
    "tleap-pipeline",
    "tcn-pipeline",
    "transformer-pipeline",
    "gnn-pipeline",
    "graph-transformer-pipeline"
  ]
}
