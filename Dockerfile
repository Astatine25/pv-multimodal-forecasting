# ===== Base image =====
FROM python:3.11-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install system dependencies efficiently =====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        wget \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ===== Copy only requirements first (cache) =====
COPY requirements.txt .

# ===== Install Python dependencies =====
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== Copy rest of the code =====
COPY . .

# ===== Expose Streamlit port =====
EXPOSE 8501

# ===== Set default command =====
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
