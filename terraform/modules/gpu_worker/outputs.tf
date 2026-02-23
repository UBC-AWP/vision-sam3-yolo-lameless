# GPU Worker Module Outputs

output "asg_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.gpu_worker.name
}

output "asg_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = aws_autoscaling_group.gpu_worker.arn
}

output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.gpu_worker.id
}

output "instance_profile_arn" {
  description = "ARN of the instance profile"
  value       = aws_iam_instance_profile.gpu_worker.arn
}

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.gpu_worker.name
}
