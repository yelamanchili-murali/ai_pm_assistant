# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the specified files into the container at /app
COPY app.py /app/
COPY LICENSE /app/
COPY README.md /app/
COPY requirements.txt /app/
COPY startup.sh /app/
COPY .env /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Define environment variable
ENV NAME World

# Run startup.sh when the container launches
CMD ["sh", "startup.sh"]