#!/bin/bash
# AWS Infrastructure Setup Script
# This script creates ECR repositories, configures Terraform, and deploys infrastructure
#
# Usage: ./setup_aws.sh [--dry-run] [--skip-ecr] [--skip-deploy]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$TERRAFORM_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
SKIP_ECR=false
SKIP_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-ecr)
            SKIP_ECR=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./setup_aws.sh [--dry-run] [--skip-ecr] [--skip-deploy]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================="
echo "  AWS Infrastructure Setup"
echo -e "==============================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    echo "Install with: brew install awscli (macOS) or pip install awscli"
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    echo -e "${RED}Error: Terraform is not installed${NC}"
    echo "Install with: brew install terraform (macOS)"
    exit 1
fi

# Verify AWS credentials
echo "Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured${NC}"
    echo "Run: aws configure"
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "us-east-1")

echo -e "${GREEN}✓ AWS Account: $AWS_ACCOUNT_ID${NC}"
echo -e "${GREEN}✓ AWS Region: $AWS_REGION${NC}"
echo ""

# ECR Registry URL
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Services that need ECR repositories
ECR_SERVICES=(
    "admin-backend"
    "admin-frontend"
    "video-ingestion"
    "video-preprocessing"
    "clip-curation"
    "tracking-service"
    "ml-pipeline"
    "fusion-service"
    "nats"
    "qdrant"
    "yolo-pipeline"
    "sam3-pipeline"
    "dinov3-pipeline"
    "tleap-pipeline"
    "tcn-pipeline"
    "transformer-pipeline"
    "gnn-pipeline"
    "graph-transformer-pipeline"
)

# ============================================
# Step 2: Create ECR Repositories
# ============================================
if [ "$SKIP_ECR" = false ]; then
    echo -e "${BLUE}=============================================="
    echo "  Step 2: Creating ECR Repositories"
    echo -e "==============================================${NC}"
    echo ""

    for SERVICE in "${ECR_SERVICES[@]}"; do
        echo -n "Creating repository: $SERVICE... "

        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY RUN]${NC}"
        else
            if aws ecr describe-repositories --repository-names "$SERVICE" &> /dev/null; then
                echo -e "${GREEN}already exists${NC}"
            else
                aws ecr create-repository \
                    --repository-name "$SERVICE" \
                    --image-scanning-configuration scanOnPush=true \
                    --encryption-configuration encryptionType=AES256 \
                    > /dev/null 2>&1
                echo -e "${GREEN}created${NC}"
            fi
        fi
    done

    echo ""
    echo -e "${GREEN}✓ ECR Registry: $ECR_REGISTRY${NC}"
    echo ""
fi

# ============================================
# Step 3: Configure Terraform Variables
# ============================================
echo -e "${BLUE}=============================================="
echo "  Step 3: Configuring Terraform Variables"
echo -e "==============================================${NC}"
echo ""

TFVARS_FILE="$TERRAFORM_DIR/terraform.tfvars"

if [ -f "$TFVARS_FILE" ]; then
    echo -e "${YELLOW}terraform.tfvars already exists${NC}"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing terraform.tfvars"
    else
        rm "$TFVARS_FILE"
    fi
fi

if [ ! -f "$TFVARS_FILE" ]; then
    # Prompt for sensitive values
    echo "Please enter the following values:"
    echo ""

    read -p "Database password (min 8 chars): " -s DB_PASSWORD
    echo ""

    if [ ${#DB_PASSWORD} -lt 8 ]; then
        echo -e "${YELLOW}Password too short, generating random password...${NC}"
        DB_PASSWORD=$(openssl rand -base64 16 | tr -dc 'a-zA-Z0-9' | head -c 16)
        echo -e "${GREEN}Generated password: $DB_PASSWORD${NC}"
        echo -e "${YELLOW}Save this password! You'll need it later.${NC}"
    fi

    read -p "JWT secret key (or press Enter to generate): " -s JWT_SECRET
    echo ""

    if [ -z "$JWT_SECRET" ]; then
        JWT_SECRET=$(openssl rand -base64 32)
        echo -e "${GREEN}Generated JWT secret${NC}"
    fi

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would create terraform.tfvars${NC}"
    else
        cat > "$TFVARS_FILE" << EOF
# Terraform Variables - Auto-generated by setup_aws.sh
# Generated on: $(date)

# General Configuration
project_name = "cow-lameness"
environment  = "production"
aws_region   = "$AWS_REGION"

# Networking
vpc_cidr = "10.0.0.0/16"

# Database
db_password = "$DB_PASSWORD"

# Security
jwt_secret = "$JWT_SECRET"

# Container Registry
ecr_registry = "$ECR_REGISTRY"

# SSL Certificate (optional - uncomment and set if you have one)
# certificate_arn = "arn:aws:acm:${AWS_REGION}:${AWS_ACCOUNT_ID}:certificate/xxx"

# GPU Worker Configuration
gpu_enabled        = false
gpu_instance_type  = "g4dn.xlarge"
use_spot_instances = true
EOF
        echo -e "${GREEN}✓ Created terraform.tfvars${NC}"
    fi
fi

echo ""

# ============================================
# Step 4: Deploy Infrastructure
# ============================================
if [ "$SKIP_DEPLOY" = false ]; then
    echo -e "${BLUE}=============================================="
    echo "  Step 4: Deploying Infrastructure"
    echo -e "==============================================${NC}"
    echo ""

    cd "$TERRAFORM_DIR"

    # Terraform init
    echo "Initializing Terraform..."
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] terraform init${NC}"
    else
        terraform init
    fi
    echo ""

    # Terraform plan
    echo "Creating execution plan..."
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] terraform plan${NC}"
    else
        terraform plan -out=tfplan
    fi
    echo ""

    # Confirm deployment
    echo -e "${YELLOW}=============================================="
    echo "  Cost Estimate"
    echo -e "==============================================${NC}"
    echo ""
    echo "Estimated monthly costs:"
    echo "  - ECS Fargate:    ~\$120"
    echo "  - NAT Gateway:    ~\$35"
    echo "  - ALB:            ~\$20"
    echo "  - RDS:            ~\$15"
    echo "  - EFS:            ~\$20"
    echo "  - Other:          ~\$13"
    echo "  ─────────────────────"
    echo "  Total:            ~\$223/month"
    echo ""
    echo "  GPU (when enabled): +\$0.16-0.52/hour"
    echo ""

    read -p "Do you want to deploy now? (y/N) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}[DRY RUN] terraform apply${NC}"
        else
            echo ""
            echo "Deploying infrastructure (this may take 10-15 minutes)..."
            terraform apply tfplan
        fi
    else
        echo ""
        echo "Deployment skipped. To deploy later, run:"
        echo "  cd terraform && terraform apply tfplan"
    fi
fi

# ============================================
# Summary
# ============================================
echo ""
echo -e "${GREEN}=============================================="
echo "  Setup Complete!"
echo -e "==============================================${NC}"
echo ""
echo "ECR Registry: $ECR_REGISTRY"
echo ""
echo "Next steps:"
echo "  1. Add these GitHub Secrets:"
echo "     - AWS_ACCESS_KEY_ID"
echo "     - AWS_SECRET_ACCESS_KEY"
echo "     - AWS_REGION: $AWS_REGION"
echo "     - ECR_REGISTRY: $ECR_REGISTRY"
echo "     - DB_PASSWORD: (from terraform.tfvars)"
echo "     - JWT_SECRET: (from terraform.tfvars)"
echo ""
echo "  2. Push code to trigger CI/CD:"
echo "     git add . && git commit -m 'Add AWS infrastructure' && git push"
echo ""
echo "  3. Enable GPU for demos:"
echo "     ./terraform/scripts/gpu_enable.sh"
echo ""
