FROM python:3.15

WORKDIR /app

# Evita file .pyc e abilita output Python immediato
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Aggiorna pip
RUN pip install --no-cache-dir --upgrade pip

# Copia metadati e codice, poi installa la libreria con le sue dipendenze
COPY pyproject.toml README.md ./
COPY to_juries ./to_juries
RUN pip install --no-cache-dir -e .

# Mantiene attivo il container
CMD ["tail", "-f", "/dev/null"]