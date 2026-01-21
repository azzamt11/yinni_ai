FROM python:3.10-slim

WORKDIR /app

# 1. Install dependencies
# Because the context is the root, we look inside the subfolder
COPY ultra_lite_classifier/requirement.txt . 
RUN pip install --no-cache-dir -r requirement.txt

# 2. Copy the Python source code from the subfolder to /app
COPY ultra_lite_classifier/ .

# 3. Copy the model files from the root context to /app
# These are the files that were missing in your previous log
COPY ulc_model_config.json .
COPY ulc_model_weights.pth .
COPY ulc_vocab.pkl .

# 4. Set environment variables
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# 5. Run the application
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]