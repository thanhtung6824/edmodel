FROM python:3.11-slim

WORKDIR /app

# Install torch CPU first (separate layer for caching)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy source code and model
COPY src/ /app/src/
COPY server/ /app/server/
COPY best_model_transformer.pth /app/best_model_transformer.pth

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
