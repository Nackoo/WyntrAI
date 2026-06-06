FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask==2.3.3 flask-cors==4.0.0 gunicorn numpy requests beautifulsoup4
EXPOSE 7860
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:7860"]
