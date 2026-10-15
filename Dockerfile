FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import duckdb; c = duckdb.connect(); c.sql('INSTALL httpfs; INSTALL h3 FROM community')"

COPY . /app
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA PORT=8000
CMD ["python", "server.py"]
