FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose Streamlit port
EXPOSE 8501

# Run the dashboard
CMD ["streamlit", "run", "src/discovery_agent/app.py", "--server.address", "0.0.0.0"]
