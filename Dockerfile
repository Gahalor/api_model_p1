# Usa Python 3.9 como base
FROM python:3.9

# Instala Java para H2O
RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*

# Crea el directorio de trabajo
WORKDIR /app

# Copia el contenido del proyecto
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 5310
EXPOSE 5310

# Lanza la aplicación Flask con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5310", "model:app"]
