FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Install CPU-only PyTorch.
# GPU inference is handled by Ollama on the Windows host.
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project components required by the agent dependency chain.
COPY week03 ./week03
COPY week06 ./week06
COPY week07 ./week07
COPY week08 ./week08
COPY week09 ./week09

EXPOSE 8000

CMD ["uvicorn", "week09.agent_api.main:app", "--host", "0.0.0.0", "--port", "8000"]