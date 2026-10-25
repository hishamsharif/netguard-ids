FROM python:3.11-slim

# System deps for torch + shap compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/       ./app/
COPY mcp_server/ ./mcp_server/
COPY ui/        ./ui/
COPY training/  ./training/

# Models and data are volume-mounted at runtime; create empty placeholders
RUN mkdir -p models data/cicids2017 data/nslkdd

# Non-root user
RUN useradd -m -u 1000 netguard && chown -R netguard:netguard /app
USER netguard

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
