# GPU Worker Module - EC2 Auto Scaling Group for GPU processing

# Get latest Deep Learning AMI
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.3.1 (Amazon Linux 2) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# IAM Role for GPU Worker
resource "aws_iam_role" "gpu_worker" {
  name = "${var.name_prefix}-gpu-worker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.name_prefix}-gpu-worker-role"
  }
}

resource "aws_iam_role_policy_attachment" "gpu_worker_ecr" {
  role       = aws_iam_role.gpu_worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "gpu_worker_ssm" {
  role       = aws_iam_role.gpu_worker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "gpu_worker" {
  name = "${var.name_prefix}-gpu-worker-policy"
  role = aws_iam_role.gpu_worker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
          "elasticfilesystem:ClientRootAccess"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "gpu_worker" {
  name = "${var.name_prefix}-gpu-worker-profile"
  role = aws_iam_role.gpu_worker.name
}

# Launch Template for GPU Worker
resource "aws_launch_template" "gpu_worker" {
  name_prefix   = "${var.name_prefix}-gpu-worker-"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.gpu_worker.arn
  }

  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [var.gpu_security_group_id]
    delete_on_termination       = true
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    efs_file_system_id = var.efs_file_system_id
    ecr_registry       = var.ecr_registry
    nats_endpoint      = var.nats_endpoint
    gpu_services       = join(" ", var.gpu_services)
    name_prefix        = var.name_prefix
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.name_prefix}-gpu-worker"
    }
  }

  tag_specifications {
    resource_type = "volume"
    tags = {
      Name = "${var.name_prefix}-gpu-worker-volume"
    }
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name = "${var.name_prefix}-gpu-worker-lt"
  }
}

# Auto Scaling Group
resource "aws_autoscaling_group" "gpu_worker" {
  name                = "${var.name_prefix}-gpu-worker-asg"
  desired_capacity    = var.gpu_enabled ? 1 : 0
  min_size            = 0
  max_size            = 1
  vpc_zone_identifier = var.private_subnet_ids

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = var.use_spot_instances ? 0 : 1
      on_demand_percentage_above_base_capacity = var.use_spot_instances ? 0 : 100
      spot_allocation_strategy                 = "lowest-price"
      on_demand_allocation_strategy            = "lowest-price"
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu_worker.id
        version            = "$Latest"
      }

      override {
        instance_type = var.gpu_instance_type
      }

      # Fallback instance types - multiple options for better availability
      override {
        instance_type = "g4dn.xlarge"
      }

      override {
        instance_type = "g4dn.2xlarge"
      }

      override {
        instance_type = "g5.2xlarge"
      }
    }
  }

  health_check_type         = "EC2"
  health_check_grace_period = 300

  tag {
    key                 = "Name"
    value               = "${var.name_prefix}-gpu-worker"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = "production"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

# CloudWatch Log Group for GPU Worker
resource "aws_cloudwatch_log_group" "gpu_worker" {
  name              = "/ec2/${var.name_prefix}-gpu-worker"
  retention_in_days = 30

  tags = {
    Name = "${var.name_prefix}-gpu-worker-logs"
  }
}
