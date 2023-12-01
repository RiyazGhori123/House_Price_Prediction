# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Flask will run on
EXPOSE 5002

# Define environment variable for Flask to run in production mode
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "main.py"]
