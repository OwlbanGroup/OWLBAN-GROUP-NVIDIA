# Live Deployment Guide for JPMorgan Financial APIs

This guide provides comprehensive steps to deploy the JPMorgan Financial APIs to a live production environment.

## Prerequisites

### AWS Account Setup
1. **AWS Account**: Ensure you have an AWS account with appropriate permissions
2. **AWS CLI**: Install and configure AWS CLI
   ```bash
   aws configure
   ```
3. **Docker**: Install Docker on your local machine
4. **Git**: Ensure you have Git installed

### Domain and SSL
1. **Domain Name**: Purchase a domain name (e.g., api.jpmorgan-financial.com)
2. **SSL Certificate**: AWS Certificate Manager will handle SSL certificates

## Deployment Options

### Option 1: AWS CloudFormation (Recommended)

#### Step 1: Prepare AWS Infrastructure
```bash
# Create VPC and subnets (if not existing)
# Note: Update the CloudFormation parameters with your VPC details
aws ec2 describe-vpcs
aws ec2 describe-subnets
```

#### Step 2: Set Environment Variables
Create a `.env.prod` file with production settings:
```bash
# Database credentials
DB_USERNAME=your_db_username
DB_PASSWORD=your_secure_db_password

# AWS Region
AWS_REGION=us-east-1

# Environment
FLASK_ENV=production
LOG_LEVEL=INFO
```

#### Step 3: Build and Push Docker Image
```bash
cd jpmorgan_financial_apis

# Build Docker image
docker build -t jpmorgan-financial-apis:latest .

# Tag for ECR (replace with your account ID)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/jpmorgan-financial-apis:latest"
docker tag jpmorgan-financial-apis:latest $ECR_URI

# Login to ECR and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker push $ECR_URI
```

#### Step 4: Deploy with CloudFormation
```bash
# Update CloudFormation parameters
STACK_NAME="jpmorgan-financial-apis-prod"
VPC_ID="vpc-xxxxxxxx"
SUBNET_IDS="subnet-xxxxxxxx,subnet-yyyyyyyy"
DB_USERNAME="telemetry_user"
DB_PASSWORD="your_secure_password"

# Deploy stack
aws cloudformation create-stack \
    --stack-name $STACK_NAME \
    --template-body file://aws-cloudformation.yml \
    --parameters \
        ParameterKey=ImageUri,ParameterValue=$ECR_URI \
        ParameterKey=VpcId,ParameterValue=$VPC_ID \
        ParameterKey=SubnetIds,ParameterValue=$SUBNET_IDS \
        ParameterKey=DBUsername,ParameterValue=$DB_USERNAME \
        ParameterKey=DBPassword,ParameterValue=$DB_PASSWORD \
        ParameterKey=Environment,ParameterValue=production \
    --capabilities CAPABILITY_IAM \
    --region us-east-1

# Wait for deployment
aws cloudformation wait stack-create-complete --stack-name $STACK_NAME --region us-east-1
```

#### Step 5: Configure Domain and SSL
```bash
# Get Load Balancer DNS
LB_DNS=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region us-east-1 --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDns`].OutputValue' --output text)

# Create Route 53 hosted zone (if not exists)
HOSTED_ZONE_ID=$(aws route53 create-hosted-zone --name api.jpmorgan-financial.com --caller-reference $(date +%s) --query 'HostedZone.Id' --output text)

# Create SSL certificate
CERT_ARN=$(aws acm request-certificate --domain-name api.jpmorgan-financial.com --validation-method DNS --query 'CertificateArn' --output text)

# Add HTTPS listener to ALB
LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn $(aws elbv2 describe-load-balancers --names ${STACK_NAME}-alb --query 'LoadBalancers[0].LoadBalancerArn' --output text) \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=$CERT_ARN \
    --default-actions Type=forward,TargetGroupArn=$(aws elbv2 describe-target-groups --names ${STACK_NAME}-tg --query 'TargetGroups[0].TargetGroupArn' --output text) \
    --query 'Listeners[0].ListenerArn' --output text)
```

#### Step 6: Update DNS Records
```bash
# Create Route 53 record
aws route53 change-resource-record-sets \
    --hosted-zone-id $HOSTED_ZONE_ID \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "api.jpmorgan-financial.com",
                "Type": "A",
                "AliasTarget": {
                    "DNSName": "'$LB_DNS'",
                    "HostedZoneId": "Z35SXDOTRQ7X7K",
                    "EvaluateTargetHealth": true
                }
            }
        }]
    }'
```

### Option 2: Kubernetes Deployment

#### Step 1: Set up Kubernetes Cluster
```bash
# Using EKS
eksctl create cluster --name jpmorgan-financial-apis --region us-east-1

# Or using kops, or other Kubernetes providers
```

#### Step 2: Deploy with Kubernetes
```bash
cd jpmorgan_financial_apis/k8s

# Update ConfigMap and Secret with production values
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# Deploy PostgreSQL and Redis
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
```

### Option 3: Docker Compose (For smaller scale)

#### Step 1: Production Docker Compose
```bash
cd jpmorgan_financial_apis

# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

## Post-Deployment Configuration

### Step 1: Health Checks
```bash
# Test health endpoint
curl https://api.jpmorgan-financial.com/health

# Expected response:
{
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "1.0.0"
}
```

### Step 2: Database Migration
```bash
# Run database migrations if needed
# The application should handle this automatically on startup
```

### Step 3: Monitoring Setup
```bash
# Enable CloudWatch monitoring
aws logs create-log-group --log-group-name /ecs/jpmorgan-financial-apis --region us-east-1

# Set up CloudWatch alarms
aws cloudwatch put-metric-alarm \
    --alarm-name "HighCPUUtilization" \
    --alarm-description "CPU utilization is high" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=ClusterName,Value=jpmorgan-financial-apis-cluster Name=ServiceName,Value=jpmorgan-financial-apis-service \
    --region us-east-1
```

### Step 4: Backup Configuration
```bash
# Enable RDS automated backups
aws rds modify-db-instance \
    --db-instance-identifier jpmorgan-financial-apis-db \
    --backup-retention-period 7 \
    --preferred-backup-window 03:00-04:00 \
    --region us-east-1
```

## Security Considerations

### 1. Network Security
- Ensure security groups only allow necessary traffic
- Use private subnets for database and Redis
- Enable VPC flow logs

### 2. Application Security
- Implement proper authentication/authorization
- Use HTTPS only
- Regular security updates
- Implement rate limiting

### 3. Data Security
- Encrypt data at rest and in transit
- Implement proper access controls
- Regular security audits

## Scaling and Performance

### Auto Scaling
```bash
# ECS Service Auto Scaling
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/jpmorgan-financial-apis-cluster/jpmorgan-financial-apis-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 1 \
    --max-capacity 10 \
    --region us-east-1

# CPU-based scaling policy
aws application-autoscaling put-scaling-policy \
    --policy-name cpu-scaling-policy \
    --service-namespace ecs \
    --resource-id service/jpmorgan-financial-apis-cluster/jpmorgan-financial-apis-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration "TargetValue=70.0,PredefinedMetricSpecification={PredefinedMetricType=ECSServiceAverageCPUUtilization}" \
    --region us-east-1
```

### Database Scaling
- Monitor RDS metrics
- Consider read replicas for read-heavy workloads
- Use ElastiCache for session storage and caching

## Monitoring and Alerting

### Application Monitoring
- CloudWatch Logs for application logs
- CloudWatch Metrics for performance metrics
- X-Ray for distributed tracing

### Infrastructure Monitoring
- EC2 instance metrics
- RDS database metrics
- ELB access logs

### Alerting
- Set up SNS topics for notifications
- Configure CloudWatch alarms
- Implement PagerDuty or similar alerting system

## Rollback Procedures

### Emergency Rollback
```bash
# Quick rollback to previous version
aws ecs update-service \
    --cluster jpmorgan-financial-apis-cluster \
    --service jpmorgan-financial-apis-service \
    --task-definition previous-task-definition-arn \
    --region us-east-1
```

### Database Rollback
- Use RDS snapshots for point-in-time recovery
- Test rollback procedures regularly

## Cost Optimization

### Reserved Instances
- Consider Reserved Instances for steady-state workloads
- Use Spot instances for development/testing

### Storage Optimization
- Monitor S3 storage costs
- Implement lifecycle policies for logs

### Cost Monitoring
```bash
# Set up cost allocation tags
aws ce create-cost-category-definition \
    --name "JPMorgan-Financial-APIs" \
    --rule-version "CostCategoryExpression.v1" \
    --rules '[{"value":"Production","rule":{"tags":{"key":"Environment","values":["production"]}}}]' \
    --region us-east-1
```

## Testing in Production

### Smoke Tests
```bash
# Run smoke tests against production
curl https://api.jpmorgan-financial.com/health
curl https://api.jpmorgan-financial.com/telemetry -X POST -H "Content-Type: application/json" -d '{"test": "data"}'
```

### Load Testing
- Use tools like Artillery or Locust for load testing
- Monitor performance under load
- Set up performance baselines

## Maintenance Procedures

### Regular Updates
1. Monitor for security updates
2. Update dependencies regularly
3. Rotate credentials periodically
4. Review and update infrastructure as code

### Backup Verification
- Regularly test backup restoration
- Verify backup integrity
- Document recovery time objectives (RTO) and recovery point objectives (RPO)

## Support and Documentation

### Runbooks
- Create detailed runbooks for common operations
- Document troubleshooting procedures
- Maintain incident response plans

### Knowledge Base
- Document architecture decisions
- Maintain API documentation
- Create onboarding guides for new team members

## Final Verification

After deployment, verify:
- [ ] Application is accessible via domain
- [ ] SSL certificate is valid
- [ ] All endpoints respond correctly
- [ ] Database connections work
- [ ] Monitoring is configured
- [ ] Backups are scheduled
- [ ] Security groups are properly configured
- [ ] Auto-scaling is working
- [ ] Cost monitoring is active

## Contact Information

For deployment issues or questions:
- DevOps Team: devops@jpmorgan.com
- Application Team: api-team@jpmorgan.com
- Security Team: security@jpmorgan.com
