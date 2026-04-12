FROM python:3.12-slim

WORKDIR /server

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run the server via the app entry point
CMD ["python", "-c", "server.app"]
