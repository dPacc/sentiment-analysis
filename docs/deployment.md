# API Deployment

This document provides comprehensive instructions for deploying the Twitter sentiment analysis model as a scalable API service.

## Overview

The deployment architecture consists of several components working together to provide a high-performance, scalable, and reliable sentiment analysis service:

1. **Flask API**: Core application that serves the sentiment analysis model through RESTful endpoints
2. **NGINX**: Load balancer and reverse proxy for handling high volumes of requests
3. **Docker**: Containerization platform for consistent deployment across environments
4. **Prometheus**: Monitoring system for collecting metrics
5. **Grafana**: Visualization platform for monitoring dashboards

This architecture is designed to handle at least 500 concurrent users, with the ability to scale horizontally as needed.

## System Requirements

### Minimum Requirements
- 4 CPU cores
- 8 GB RAM
- 20 GB disk space
- NVIDIA GPU with at least 4GB VRAM (for GPU inference)

### Recommended Requirements
- 8+ CPU cores
- 16+ GB RAM
- 50+ GB SSD
- NVIDIA GPU with 8+ GB VRAM (Tesla T4, V100, or newer)
- Docker and Docker Compose
- NVIDIA Container Toolkit (for GPU support)

## Deployment Options

### 1. Docker Compose Deployment (Recommended)

Docker Compose provides an easy way to deploy all components together with proper configuration.

#### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Container Toolkit (for GPU support)
- Trained model file (`best_model.pt`)

#### Steps

1. **Prepare the model file**:
   
   Ensure your trained model file (`best_model.pt`) is placed in the `models` directory:
   ```bash
   mkdir -p models
   cp /path/to/your/best_model.pt models/
   ```

2. **Configure environment variables**:
   
   Create a `.env` file in the project root:
   ```
   MODEL_PATH=/app/models/best_model.pt
   PORT=5000
   HOST=0.0.0.0
   DEVICE=cuda  # Use 'cpu' if not using GPU
   ```

3. **Start the services**:
   
   ```bash
   docker-compose up -d
   ```
   
   This command will build and start all services defined in the `docker-compose.yml` file.

4. **Verify deployment**:
   
   Check if all services are running:
   ```bash
   docker-compose ps
   ```
   
   Test the API health endpoint:
   ```bash
   curl http://localhost/api/health
   ```
   
   You should receive a response like:
   ```json
   {"status": "healthy", "message": "API is running"}
   ```

### 2. Manual Deployment

If you prefer to deploy components individually or don't want to use Docker, you can deploy the API manually.

#### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (for GPU inference)
- Trained model file (`best_model.pt`)

#### Steps

1. **Install dependencies**:
   
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server**:
   
   ```bash
   python -m src.api.app --model_path ./models/best_model.pt --host 0.0.0.0 --port 5000
   ```

3. **Configure NGINX (optional)**:
   
   If you want to use NGINX as a reverse proxy, install it and use the provided configuration file:
   
   ```bash
   sudo cp nginx/conf.d/default.conf /etc/nginx/conf.d/
   sudo systemctl restart nginx
   ```

## Scaling for Production

### Horizontal Scaling

To handle more traffic, you can scale the API service horizontally by adding more instances:

1. **Update the NGINX configuration**:
   
   Add more server entries to the upstream block in `nginx/conf.d/default.conf`:
   
   ```
   upstream api_servers {
       server api1:5000;
       server api2:5000;
       server api3:5000;
       # Add more servers as needed
   }
   ```

2. **Update Docker Compose configuration**:
   
   Add more API service instances in `docker-compose.yml`:
   
   ```yaml
   api1:
     build:
       context: .
       dockerfile: Dockerfile
     # ...
   
   api2:
     build:
       context: .
       dockerfile: Dockerfile
     # ...
   
   api3:
     build:
       context: .
       dockerfile: Dockerfile
     # ...
   ```

   Alternatively, use Docker Compose's scale option:
   
   ```bash
   docker-compose up -d --scale api=3
   ```

### Performance Optimization

To optimize the performance of the API service:

1. **Use GPU acceleration**:
   
   Ensure the `DEVICE` environment variable is set to `cuda` and the Docker container has access to GPU resources.

2. **Adjust batch processing parameters**:
   
   In `src/api/app.py`, you can adjust the `max_workers` variable for batch processing:
   
   ```python
   max_workers = 20  # Increase for more parallel processing
   ```

3. **Optimize model size**:
   
   Consider using model quantization or distillation to reduce model size and increase inference speed.

4. **Implement caching**:
   
   Add a caching layer for frequently requested texts to reduce computation.

## Monitoring and Maintenance

### Prometheus Monitoring

The API service is instrumented with Prometheus metrics. Access the Prometheus dashboard at:

```
http://localhost:9090
```

Key metrics to monitor:
- Request rate
- Response time
- Error rate
- Memory usage
- GPU utilization (if using GPU)

### Grafana Dashboards

Grafana provides visualization for Prometheus metrics. Access the Grafana dashboard at:

```
http://localhost:3000
```

Default credentials:
- Username: admin
- Password: admin

A pre-configured dashboard is available for the sentiment analysis API, showing key performance metrics.

### Log Management

Logs from all services are collected and can be viewed using Docker:

```bash
docker-compose logs -f
```

To view logs for a specific service:

```bash
docker-compose logs -f api
```

### Backup and Restore

To backup the trained model:

```bash
cp models/best_model.pt /path/to/backup/location/
```

To restore from backup:

```bash
cp /path/to/backup/location/best_model.pt models/
```

## Security Considerations

To enhance the security of your deployment:

1. **Enable HTTPS**:
   
   Update the NGINX configuration to enable HTTPS with SSL certificates.

2. **Implement authentication**:
   
   Add API key authentication to the Flask application.

3. **Rate limiting**:
   
   The provided NGINX configuration includes rate limiting, but you may need to adjust the limits based on your requirements.

4. **Input validation**:
   
   Ensure all API inputs are properly validated to prevent injection attacks.

## Troubleshooting

### Common Issues

1. **API service fails to start**:
   
   - Check if the model file exists at the specified path
   - Ensure GPU is available if using CUDA
   - Check for sufficient memory
   
   Solution: Adjust environment variables or switch to CPU mode.

2. **High response times**:
   
   - Check system resource utilization
   - Monitor batch sizes
   - Verify network latency
   
   Solution: Scale horizontally or optimize configurations.

3. **Out of memory errors**:
   
   - Reduce batch size
   - Use smaller model variant
   - Add more RAM or GPU memory
   
   Solution: Adjust resource allocations in Docker Compose.

### Support Resources

For additional help:
- Check the project GitHub repository
- Consult the PyTorch and Hugging Face documentation
- Review the Docker and NGINX documentation

## Updating the Model

To update the deployed model with a new version:

1. **Train new model**:
   
   Follow the training process to create a new model.

2. **Replace model file**:
   
   ```bash
   cp /path/to/new/model.pt models/best_model.pt
   ```

3. **Restart the API service**:
   
   ```bash
   docker-compose restart api
   ```

This hot-swap approach minimizes downtime during model updates.