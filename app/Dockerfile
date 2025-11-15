# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . .

# Make port 7860 available to the world outside this container
# Hugging Face Spaces expects the app to run on this port by default
EXPOSE 7860

# (Optional) Set environment variables like FLASK_APP if you need to, but not strictly required
# ENV FLASK_APP=app.py

# Use gunicorn to run the app (use gevent worker for async handling)
# --workers 1: Adjust based on your app's needs (use 2+ for concurrency, but 1 is fine for small apps)

VOLUME [ "/app/examples", "/app/data", "/app/data/uploads", "/app/uploads" ]

CMD ["gunicorn", "--worker-class", "gevent", "--workers", "1", "--timeout", "120", "--bind", "0.0.0.0:7860", "--log-level", "debug", "app:app"]
