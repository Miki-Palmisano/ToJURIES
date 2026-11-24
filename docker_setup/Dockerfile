FROM python:3.15

WORKDIR /app

# Evita file .pyc e abilita output Python immediato
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Aggiorna pip
RUN pip install --no-cache-dir --upgrade pip

# Copia prima requirements.txt per sfruttare la cache Docker
COPY requirements.txt /app/requirements.txt

# Installa le dipendenze
RUN pip install --no-cache-dir -r /app/requirements.txt

# Mantiene attivo il container
CMD ["tail", "-f", "/dev/null"]