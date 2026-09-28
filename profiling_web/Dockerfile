FROM node:22-bookworm-slim AS frontend
WORKDIR /ui
RUN npm install -g pnpm@11.25.0
COPY profiling_web/frontend/package.json profiling_web/frontend/pnpm-lock.yaml profiling_web/frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY profiling_web/frontend/ ./
RUN pnpm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PROFILING_WEB_DATA=/data
WORKDIR /app
COPY profiling_web/requirements.lock.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && useradd --uid 10001 --create-home profiler && mkdir /data && chown profiler:profiler /data
COPY profiling_tools/*.py /app/profiling_tools/
COPY profiling_web/*.py /app/profiling_web/
COPY --from=frontend /ui/dist /app/profiling_web/frontend/dist
USER profiler
EXPOSE 8000
CMD ["python", "-m", "profiling_web", "serve", "--host", "0.0.0.0"]
