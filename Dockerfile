# Use an official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

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
