FROM python:3.10-slim

WORKDIR /app

# Install dependencies
# Docker looks for this file inside the 'context' defined in compose
COPY requirement.txt . 
RUN pip install --no-cache-dir -r requirement.txt

# Copy all files from the context (ultra_lite_classifier folder) to /app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]