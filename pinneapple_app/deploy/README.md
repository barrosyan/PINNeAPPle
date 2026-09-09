# Deploying `pinneapple_app` to Kubernetes

`pinneapple_app`'s own `docker-compose.yml` (one level up) is the
single-machine story. This directory adds the multi-node one: raw `k8s/`
manifests, a `helm/pinneapple-app` chart wrapping them, and a
Prometheus/Grafana `monitoring/` setup. Nothing here replaces the
Compose file — build the same two images (`Dockerfile.backend`,
`Dockerfile.frontend`, one level up) and push them to a registry your
cluster can pull from first.

## Plain manifests

```
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl create secret generic pinneapple-admin --from-literal=token=<a-real-secret> -n pinneapple
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml   # requires an nginx ingress controller
```

## Helm chart

```
helm install pinneapple-app helm/pinneapple-app \
  --set backend.image.tag=<your-tag> \
  --set frontend.image.tag=<your-tag>
```

The admin token Secret (`pinneapple-admin`, key `token`) is **not**
created by the chart — create it yourself first, same as with the plain
manifests. Without it, `/api/admin/*` returns `503` rather than opening
up (see `backend/core/admin_auth.py`).

## Monitoring

`monitoring/prometheus.yml` scrapes the backend's `/api/metrics`
(Prometheus text format — see `backend/main.py`, a hand-rolled two-gauge
exposition, no new dependency). `monitoring/grafana-datasource.yaml`
wires a Grafana instance to that Prometheus. Both are plain config files,
not wired into the Helm chart or a bundled Prometheus/Grafana
deployment — bring your own Prometheus/Grafana install (or the community
Helm charts for them) and point it at these.

## What this does NOT include

- CI/CD (image build+push) — bring your own.
- TLS (the Ingress's `tls:` block is commented out/disabled by default,
  same honest "not configured out of the box" choice this repo's own
  `nginx.conf` makes).
- A bundled Prometheus/Grafana server — only the scrape/datasource
  config for one you run yourself.
