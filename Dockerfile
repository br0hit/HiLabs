# Start from minimal Ubuntu
FROM ubuntu:22.04

# Prevent tzdata interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    curl wget git nano vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /app/venv

# Activate venv and install Python packages (except numpy)
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install \
      opencv-python \
      pytesseract \
      pdf2image \
      matplotlib \
      pandas \
      PyMuPDF

# Force reinstall numpy < 2 at the end
RUN /app/venv/bin/pip install "numpy<2" --force-reinstall

# Ensure venv Python is used by default
ENV PATH="/app/venv/bin:$PATH"

# Print numpy version during build
RUN python -c "import numpy as np; print('✅ Final NumPy version:', np.__version__)"

# 💡 NEW LINE: Copy your script into the working directory (/app)
COPY pipeline_extract_clauses.py .
COPY config.json .
# COPY compare_clauses.py .
# COPY keywords.txt .
# COPY condition_words.txt .

# Default command (CMD is typically used to define the entry point, 
# but we'll override this with the `docker run` command.)
CMD ["/bin/bash"]