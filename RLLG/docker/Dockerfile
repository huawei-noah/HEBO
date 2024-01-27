# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:4.10.3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a new Conda environment
RUN conda create --name rllg python=3.8

# Activate the Conda environment
SHELL ["conda", "run", "-n", "rllg", "/bin/bash", "-c"]

# Install the package and its dependencies using setup.py
RUN pip install -e .

# Install gym
RUN pip install gym==0.21.0

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME rllg

# Run main.py when the container launches
# CMD ["conda", "run", "-n", "rllg", "python", "main.py"]
