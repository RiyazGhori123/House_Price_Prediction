# Use a base image with Python and other dependencies
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Pin Flask and Werkzeug versions
RUN pip install Flask==1.1.4 Werkzeug==1.0.1

# Upgrade pip and install required packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple/ --default-timeout=100

# Copy the rest of the application code into the container at /app
COPY . /app/

# Expose the port that Flask will run on
EXPOSE 5000

# Define environment variable
ENV NAME World

# Command to run the application
CMD ["python", "main.py"]
