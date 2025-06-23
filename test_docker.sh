#!/bin/bash

# Docker Test Script for Hospital Readmission MLOps Pipeline
# This script tests Docker functionality and provides setup guidance

set -e

echo "🐳 Testing Docker Setup for Hospital Readmission MLOps Pipeline"
echo "================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    echo "   On macOS: Start Docker Desktop"
    echo "   On Linux: sudo systemctl start docker"
    exit 1
fi

echo "✅ Docker is installed and running"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is available"
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    echo "✅ Docker Compose (v2) is available"
    COMPOSE_CMD="docker compose"
else
    echo "⚠️  Docker Compose not found. Some features may not work."
    COMPOSE_CMD=""
fi

# Test Docker build
echo ""
echo "🔨 Testing Docker build..."
if docker build -t hospital-readmission-mlops .; then
    echo "✅ Docker build successful"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Test Docker run (brief test)
echo ""
echo "🚀 Testing Docker run (brief test)..."
CONTAINER_ID=$(docker run -d -p 8000:8000 --name mlops-test hospital-readmission-mlops)

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 10

# Test health endpoint
echo ""
echo "🏥 Testing health endpoint..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ Health endpoint is working"
else
    echo "❌ Health endpoint failed"
fi

# Test API docs
echo ""
echo "📚 Testing API documentation..."
if curl -f http://localhost:8000/docs &> /dev/null; then
    echo "✅ API documentation is accessible"
else
    echo "❌ API documentation failed"
fi

# Clean up test container
echo ""
echo "🧹 Cleaning up test container..."
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo ""
echo "🎉 Docker setup test completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run the full pipeline: docker run -p 8000:8000 hospital-readmission-mlops"
echo "2. Use Docker Compose: $COMPOSE_CMD up -d"
echo "3. Test the API: curl http://localhost:8000/health"
echo "4. View documentation: http://localhost:8000/docs"
echo ""
echo "📖 For more information, see the README.md file" 