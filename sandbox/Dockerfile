# Image used to execute agent-generated Python code.
# Build: docker build -t discovery-sandbox:latest sandbox/
# Runtime hardening (no network, read-only rootfs, dropped caps, limits) is applied
# by src/discovery_agent/sandbox/runner.py, not here.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Unprivileged user; code is mounted read-only at /workspace, outputs go to /output
RUN useradd --create-home --uid 10001 runner && mkdir -p /workspace /output \
    && chown runner:runner /output
USER runner
WORKDIR /workspace

ENTRYPOINT ["python", "-I"]
CMD ["/workspace/main.py"]
