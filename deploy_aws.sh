#!/bin/bash

# AWS Cloud Deployment Script for JPMorgan Financial APIs
# This script deploys the application to AWS using ECS/Fargate

set -e

echo "☁️ Starting AWS Cloud Deployment for JPMorgan Financial APIs..."

# Check if AWS CLI is installed and configured
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install AWS CLI and configure credentials."
    exit 1
fi

# Check AWS configuration
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

echo "✅ AWS CLI configured"

# Set variables
STACK_NAME="jpmorgan-financial-apis"
REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🔧 Using AWS Account: $ACCOUNT_ID, Region: $REGION"

# Create ECR repository if it doesn't exist
echo "📦 Creating ECR repository..."
aws ecr describe-repositories --repository-names jpmorgan-financial-apis --region $REGION &> /dev/null || \
aws ecr create-repository --repository-name jpmorgan-financial-apis --region $REGION

# Get ECR login token and login
echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and tag Docker image
echo "🏗️ Building Docker image..."
docker build -t jpmorgan-financial-apis:latest .

# Tag for ECR
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/jpmorgan-financial-apis:latest"
docker tag jpmorgan-financial-apis:latest $ECR_URI

# Push to ECR
echo "📤 Pushing image to ECR..."
docker push $ECR_URI

# Create CloudFormation stack
echo "☁️ Creating CloudFormation stack..."

# Check if stack exists
if aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION &> /dev/null; then
    echo "📝 Updating existing CloudFormation stack..."
    OPERATION="update-stack"
else
    echo "🆕 Creating new CloudFormation stack..."
    OPERATION="create-stack"
fi

# Deploy CloudFormation template
aws cloudformation $OPERATION \
    --stack-name $STACK_NAME \
    --template-body file://aws-cloudformation.yml \
    --parameters ParameterKey=ImageUri,ParameterValue=$ECR_URI \
    --capabilities CAPABILITY_IAM \
    --region $REGION

# Wait for stack creation/update to complete
echo "⏳ Waiting for CloudFormation stack deployment..."
aws cloudformation wait stack-${OPERATION//-stack/}-complete --stack-name $STACK_NAME --region $REGION

# Get stack outputs
echo "📋 Getting stack outputs..."
API_URL=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)
LOAD_BALANCER_DNS=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $REGION --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDns`].OutputValue' --output text)

echo ""
echo "🎉 AWS Cloud Deployment completed successfully!"
echo ""
echo "🌐 Service URLs:"
echo "   API Gateway:     $API_URL"
echo "   Load Balancer:   http://$LOAD_BALANCER_DNS"
echo "   Health Check:    $API_URL/health"
echo ""
echo "📊 Monitoring:"
echo "   CloudWatch Logs: aws logs tail /ecs/jpmorgan-financial-apis --follow --region $REGION"
echo "   ECS Service:     aws ecs describe-services --cluster jpmorgan-financial-apis-cluster --services jpmorgan-financial-apis-service --region $REGION"
echo ""
echo "🔧 Management Commands:"
echo "   Update service:  ./deploy_aws.sh (rerun this script)"
echo "   View logs:       aws logs tail /ecs/jpmorgan-financial-apis --region $REGION"
echo "   Scale service:   aws ecs update-service --cluster jpmorgan-financial-apis-cluster --service jpmorgan-financial-apis-service --desired-count 2 --region $REGION"
echo ""
echo "💰 Cost Monitoring:"
echo "   View costs:      aws ce get-cost-and-usage --time-period Start=$(date -u +%Y-%m-01),End=$(date -u +%Y-%m-%d) --metrics BlendedCost --group-by Type=DIMENSION,Key=SERVICE --region $REGION"
