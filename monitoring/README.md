# Monitoring Setup

This project now includes a Prometheus + Grafana monitoring stack for the FastAPI application.

## What is included

- `docker-compose.yml` to start the FastAPI app, Prometheus, and Grafana together
- `monitoring/prometheus/prometheus.yml` for Prometheus scraping
- `monitoring/grafana/provisioning/datasources/datasource.yml` for auto-configured Prometheus datasource
- `monitoring/grafana/provisioning/dashboards/dashboard.yml` for auto-loading dashboards
- `monitoring/grafana/dashboards/fastapi_metrics.json` sample Grafana dashboard
- `app.py` updated to expose `/metrics`

## Start monitoring

From the project root:

```bash
docker compose up --build
```

Then open:

- FastAPI: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- MLflow UI: `http://localhost:5000`

Grafana default login:

- user: `admin`
- password: `admin`

## Metrics endpoints

- Application metrics: `http://localhost:8000/metrics`
- Health check: `http://localhost:8000/`

## Notes

- The Grafana dashboard is provisioned automatically from `monitoring/grafana/dashboards/fastapi_metrics.json`.
- Prometheus scrapes the FastAPI app every 15 seconds.
