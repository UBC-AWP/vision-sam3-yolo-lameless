# Storage Module Outputs

output "efs_file_system_id" {
  description = "ID of the EFS file system"
  value       = aws_efs_file_system.main.id
}

output "efs_file_system_arn" {
  description = "ARN of the EFS file system"
  value       = aws_efs_file_system.main.arn
}

output "efs_access_point_id" {
  description = "ID of the main EFS access point"
  value       = aws_efs_access_point.data.id
}

output "efs_videos_access_point_id" {
  description = "ID of the videos EFS access point"
  value       = aws_efs_access_point.videos.id
}

output "efs_results_access_point_id" {
  description = "ID of the results EFS access point"
  value       = aws_efs_access_point.results.id
}

output "efs_models_access_point_id" {
  description = "ID of the models EFS access point"
  value       = aws_efs_access_point.models.id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for backups"
  value       = aws_s3_bucket.backups.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for backups"
  value       = aws_s3_bucket.backups.arn
}

# Video storage outputs
output "videos_bucket_name" {
  description = "Name of the S3 bucket for videos"
  value       = aws_s3_bucket.videos.id
}

output "videos_bucket_arn" {
  description = "ARN of the S3 bucket for videos"
  value       = aws_s3_bucket.videos.arn
}

output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution for videos"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.videos[0].id : ""
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = var.enable_cloudfront ? aws_cloudfront_distribution.videos[0].domain_name : ""
}
