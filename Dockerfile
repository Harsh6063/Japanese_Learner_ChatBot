# -------- BASE IMAGE --------
FROM python:3.10-slim

# -------- SET WORKDIR --------
WORKDIR /app

# -------- SYSTEM DEPENDENCIES --------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------- COPY FILES --------
COPY . .

# -------- INSTALL PYTHON DEPENDENCIES --------
RUN pip install --no-cache-dir --upgrade pip

# (you can also use requirements.txt if you have)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------- EXPOSE PORT --------
EXPOSE 8501

# -------- RUN APP --------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]