# Observability Stack (PLG)

## Stack Components

- **Promtail**: Log collector that scrapes logs from `/var/log/supervisor/`
- **Loki**: Log aggregation system (stores logs with labels)
- **Grafana**: Visualization and dashboards (port 3001)

## Quick Start

### 1. Start the Stack

```bash
cd /app/monitoring
docker-compose up -d
```

### 2. Check Status

```bash
docker-compose ps
```

### 3. Access Grafana

- URL: http://localhost:3001
- Username: `admin`
- Password: `admin`

### 4. View Logs

In Grafana:
1. Go to "Explore" (compass icon)
2. Select "Loki" datasource
3. Query examples:
   - All backend logs: `{job="backend"}`
   - Error logs: `{job="backend"} |= "ERROR"`
   - Specific level: `{job="backend", level="error"}`
   - JSON field search: `{job="backend"} | json | level="error"`

## Log Labels

### Backend Logs
- `job`: backend
- `service`: genexus-ai-backend
- `level`: info, warning, error, debug
- `logger`: Logger name
- `module`: Python module name

### Frontend Logs
- `job`: frontend
- `service`: genexus-ai-frontend
- `level`: Extracted log level

### MongoDB Logs
- `job`: mongodb
- `service`: mongodb
- `level`: Log level
- `component`: MongoDB component

## Useful Commands

### Stop Stack
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f promtail
docker-compose logs -f loki
```

### Restart Services
```bash
docker-compose restart
```

### Clean Data
```bash
docker-compose down -v  # Removes volumes
```

## Query Examples

### Backend Errors Only
```logql
{job="backend"} | json | level="error"
```

### Chat Endpoint Logs
```logql
{job="backend"} |~ "/api/chat"
```

### Rate Limit Events
```logql
{job="backend"} |~ "429|rate limit|quota"
```

### Ingestion Events
```logql
{job="backend"} |~ "ingest"
```

### Time-based Query (Last 5 minutes)
```logql
{job="backend"} | json | level="error" [5m]
```

## Architecture

```
Host Machine (Supervisor)
    |
    ├── Backend (FastAPI) → JSON logs → /var/log/supervisor/backend.*.log
    ├── Frontend (React) → Plain logs → /var/log/supervisor/frontend.*.log
    └── MongoDB → Plain logs → /var/log/mongodb.*.log
                                    |
                                    v
                            Docker Container (Promtail)
                                    |
                                    v (push logs)
                            Docker Container (Loki)
                                    |
                                    v (query logs)
                            Docker Container (Grafana:3001)
```

## Troubleshooting

### Promtail not collecting logs

1. Check volume mount:
```bash
docker exec promtail ls -la /var/log/supervisor/
```

2. Check Promtail logs:
```bash
docker-compose logs promtail
```

### Loki not receiving logs

1. Test Loki API:
```bash
curl http://localhost:3100/ready
```

2. Check Loki logs:
```bash
docker-compose logs loki
```

### Grafana can't connect to Loki

1. Verify network:
```bash
docker network inspect monitoring_monitoring
```

2. Test from Grafana container:
```bash
docker exec grafana wget -O- http://loki:3100/ready
```

## Performance Tips

1. **Limit retention**: Loki config has 7-day retention by default
2. **Filter logs**: Use specific job labels to reduce query load
3. **Increase limits**: Edit `loki-config.yaml` if ingestion is rejected

## Security Notes

- Default Grafana credentials: `admin/admin` (change in production!)
- Loki has no authentication enabled (auth_enabled: false)
- For production: Enable auth, use secrets, configure TLS
