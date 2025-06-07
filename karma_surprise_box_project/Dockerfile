# Dockerfile

FROM python:3.10-slim

# Echo to show build progress
RUN echo "🔧 Docker build started..."

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Echo after install
RUN echo "✅ Requirements installed..."

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
