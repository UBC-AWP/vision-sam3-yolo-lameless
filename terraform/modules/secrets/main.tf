# Secrets Module - AWS Secrets Manager

# Database credentials secret
resource "aws_secretsmanager_secret" "database" {
  name                    = "${var.name_prefix}/database"
  description             = "Database credentials for cow lameness detection platform"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.name_prefix}-database-secret"
  }
}

resource "aws_secretsmanager_secret_version" "database" {
  secret_id = aws_secretsmanager_secret.database.id
  secret_string = jsonencode({
    username = "postgres"
    password = var.db_password
    host     = var.db_endpoint
    port     = 5432
    dbname   = var.db_name
    url      = "postgresql://postgres:${var.db_password}@${var.db_endpoint}/${var.db_name}"
  })
}

# JWT secret
resource "aws_secretsmanager_secret" "jwt" {
  name                    = "${var.name_prefix}/jwt"
  description             = "JWT secret for authentication"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.name_prefix}-jwt-secret"
  }
}

resource "aws_secretsmanager_secret_version" "jwt" {
  secret_id = aws_secretsmanager_secret.jwt.id
  secret_string = jsonencode({
    secret_key = var.jwt_secret
  })
}

# Application configuration secret
resource "aws_secretsmanager_secret" "app_config" {
  name                    = "${var.name_prefix}/app-config"
  description             = "Application configuration"
  recovery_window_in_days = 7

  tags = {
    Name = "${var.name_prefix}-app-config-secret"
  }
}

resource "aws_secretsmanager_secret_version" "app_config" {
  secret_id = aws_secretsmanager_secret.app_config.id
  secret_string = jsonencode({
    DATABASE_URL = "postgresql://postgres:${var.db_password}@${var.db_endpoint}/${var.db_name}"
    JWT_SECRET   = var.jwt_secret
    NATS_URL     = "nats://nats.${var.name_prefix}.local:4222"
    QDRANT_URL   = "http://qdrant.${var.name_prefix}.local:6333"
  })
}
