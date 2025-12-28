# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy repo
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose dashboard port
EXPOSE 8501

# Default command to run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
