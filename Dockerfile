FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps needed for building some wheels (e.g., cryptography).
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project files.
COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts

# Install in non-editable mode for runtime.
RUN pip install --upgrade pip && pip install .

EXPOSE 4000

CMD ["uvicorn", "cbio_nav_agent.api:app", "--host", "0.0.0.0", "--port", "5000"]
