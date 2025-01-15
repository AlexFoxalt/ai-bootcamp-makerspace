FROM python:3.11.11-slim

# Create a non-root user and set up environment variables
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy and install dependencies
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the command to run the application
ENV PATH_TO_FILE="01_Prompt Engineering and Prototyping Best Practices/app_v2.py"
CMD ["chainlit", "$PATH_TO_FILE", "--port", "7860"]
