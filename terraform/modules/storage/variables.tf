# Storage Module Variables

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_subnet_ids" {
  description = "IDs of private subnets for EFS mount targets"
  type        = list(string)
}

variable "efs_security_group" {
  description = "Security group ID for EFS"
  type        = string
}

variable "enable_cloudfront" {
  description = "Whether to create CloudFront distribution for video streaming"
  type        = bool
  default     = false
}
