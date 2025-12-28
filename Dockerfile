# -----------------------------
# Optimized Dockerfile
# -----------------------------

# 1️⃣ Base image
FROM python:3.11-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy only requirements first (leverage caching)
COPY requirements.txt .

# 5️⃣ Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy rest of repo
COPY . .

# 7️⃣ Expose Streamlit port
EXPOSE 8501

# 8️⃣ Correct CMD to run Streamlit
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
