# # Use a base image with Python and other dependencies
# FROM python:3.12

# # Set the working directory in the container
# WORKDIR /app

# # Copy the requirements file into the container at /app
# # COPY requirements.txt /app/
# COPY ./requirements.txt /app
# COPY . .
# EXPOSE 5000
# ENV FLASK_APP=my_flask.py
# # Upgrade pip and install required packages
# RUN pip install --no-cache-dir -r requirements.txt 

# # Copy the rest of the application code into the container at /app
# # COPY .. /app/

# # Expose the port that Flask will run on
# # EXPOSE 80

# # Define environment variable
# # ENV NAME World

# # Command to run the application
# # CMD ["python", "PycharmProjects/main.py"]
# CMD ["flask", "run", "--host", "0.0.0.0"]


FROM python:3.9-slim-buster
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=main.py
CMD ["flask", "run", "--host", "0.0.0.0"]