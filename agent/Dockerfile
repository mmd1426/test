FROM python:3.11.14

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy application code
COPY . .

RUN python agent.py download-files

# Run the agent
CMD ["python", "agent.py", "start"]