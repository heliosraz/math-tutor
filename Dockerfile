FROM python:3.13-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# so flask app runs on port 5000
EXPOSE 5000

CMD ["python", "./app.py"]