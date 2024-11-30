# Use an official Python base image
FROM python:3.9-slim

# Can Use an optimized FastAPI base image
#FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy the entire project directory
COPY . .

# Set the working directory to the api folder
WORKDIR /app/api

# Expose the port your API will run on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]

# Expose the port your API will run on (default for uvicorn)
#EXPOSE 80

# Command to run the application with Uvicorn
#CMD ["uvicorn", "api.main:main_app", "--host", "0.0.0.0", "--port", "80"]
