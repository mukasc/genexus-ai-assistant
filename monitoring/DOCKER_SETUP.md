# Docker Setup for PLG Stack

## Prerequisites

The monitoring stack requires Docker and Docker Compose to be installed on your system.

## Installing Docker

### Ubuntu/Debian

```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### Add current user to docker group (optional)

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Starting the Monitoring Stack

Once Docker is installed:

```bash
cd /app/monitoring
docker compose up -d
```

## Verifying the Stack

```bash
# Check running containers
docker compose ps

# Check logs
docker compose logs -f

# Access Grafana
# Open http://localhost:3001
# Login: admin/admin
```

## Stopping the Stack

```bash
cd /app/monitoring
docker compose down
```

## Troubleshooting

### Permission denied

If you get permission errors:

```bash
sudo docker compose up -d
```

### Port already in use

If port 3001 is in use, edit `docker-compose.yml`:

```yaml
grafana:
  ports:
    - "3002:3000"  # Change to different port
```

### Logs not appearing

1. Verify Promtail can access logs:
```bash
docker exec promtail ls -la /var/log/supervisor/
```

2. Check Promtail is running:
```bash
docker compose logs promtail
```

3. Test Loki API:
```bash
curl http://localhost:3100/ready
```
