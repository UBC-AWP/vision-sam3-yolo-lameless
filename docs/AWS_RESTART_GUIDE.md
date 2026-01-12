# AWS Infrastructure Restart Guide

## Current State (Shutdown)
- ECS Services: Scaled to 0
- RDS Database: Stopped
- NAT Gateway: Deleted
- GPU Worker: Not running

**Estimated cost while shutdown: ~$25-30/month** (ALB + S3 + EFS only)

---

## Quick Restart Commands

### Step 1: Recreate NAT Gateway (Required for ECS)
```bash
cd /Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/terraform
terraform apply -target=module.networking.aws_nat_gateway.main -target=module.networking.aws_eip.nat -auto-approve
```

### Step 2: Start RDS Database
```bash
aws rds start-db-instance --db-instance-identifier cow-lameness-production-postgres --region us-west-2
# Wait 5-10 minutes for RDS to be available
aws rds wait db-instance-available --db-instance-identifier cow-lameness-production-postgres --region us-west-2
```

### Step 3: Scale Up ECS Services
```bash
for service in nats qdrant admin-backend admin-frontend video-ingestion video-preprocessing clip-curation tracking-service ml-pipeline fusion-service annotation-renderer; do
  aws ecs update-service --cluster cow-lameness-production-cluster --service $service --desired-count 1 --region us-west-2
done
```

### Step 4: Start GPU Worker (After GPU images are built)
```bash
aws autoscaling set-desired-capacity --auto-scaling-group-name cow-lameness-production-gpu-worker-asg --desired-capacity 1 --region us-west-2
```

---

## Full Restart Script

Save this as `restart-aws.sh` and run it:

```bash
#!/bin/bash
set -e

echo "=== Restarting AWS Infrastructure ==="

# Step 1: NAT Gateway
echo "Step 1: Creating NAT Gateway..."
cd /Users/mehmetimga/ai-campions/vision-sam3-yolo-lameless/terraform
terraform apply -target=module.networking.aws_nat_gateway.main -target=module.networking.aws_eip.nat -auto-approve

# Step 2: RDS
echo "Step 2: Starting RDS..."
aws rds start-db-instance --db-instance-identifier cow-lameness-production-postgres --region us-west-2
echo "Waiting for RDS to be available (5-10 min)..."
aws rds wait db-instance-available --db-instance-identifier cow-lameness-production-postgres --region us-west-2
echo "RDS is ready!"

# Step 3: ECS Services
echo "Step 3: Scaling up ECS services..."
for service in nats qdrant admin-backend admin-frontend video-ingestion video-preprocessing clip-curation tracking-service ml-pipeline fusion-service annotation-renderer; do
  aws ecs update-service --cluster cow-lameness-production-cluster --service $service --desired-count 1 --region us-west-2 --query "service.serviceName" --output text
done

echo ""
echo "=== Restart Complete ==="
echo "Application URL: https://cow-lameness-production-alb-1274934122.us-west-2.elb.amazonaws.com"
echo ""
echo "Note: GPU worker is NOT started. Run this after GPU images are built:"
echo "aws autoscaling set-desired-capacity --auto-scaling-group-name cow-lameness-production-gpu-worker-asg --desired-capacity 1 --region us-west-2"
```

---

## Before Restarting - Build GPU Images

On your NVIDIA Spark box, build and push these images to ECR:
- yolo-pipeline
- sam3-pipeline
- dinov3-pipeline
- tleap-pipeline
- tcn-pipeline
- transformer-pipeline
- gnn-pipeline
- graph-transformer-pipeline

ECR Registry: `703582588105.dkr.ecr.us-west-2.amazonaws.com`

---

## Important Notes

1. **RDS Auto-Start**: AWS automatically restarts stopped RDS instances after 7 days. If you need longer shutdown, stop it again.

2. **NAT Gateway**: Must be recreated before ECS services can start (they need internet access for ECR).

3. **Terraform State**: All infrastructure is managed by Terraform. The state file knows about the deleted NAT Gateway.

4. **GPU Worker**: Only start after GPU images are pushed to ECR.

---

## Cost Summary

| State | Monthly Cost |
|-------|-------------|
| Full Shutdown | ~$25-30 |
| ECS Only (no GPU) | ~$550 |
| Full (with GPU) | ~$800-1000 |
