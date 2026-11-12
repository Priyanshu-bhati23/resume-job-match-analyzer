# Stage 1: build the React app
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

# Stage 2: API that also serves the built frontend
FROM python:3.12-slim
WORKDIR /srv/backend
COPY backend/pyproject.toml ./
COPY backend/app ./app
RUN pip install --no-cache-dir . && useradd --create-home runner
COPY --from=frontend /frontend/dist /srv/frontend/dist
ENV FRONTEND_DIST=/srv/frontend/dist
USER runner
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
