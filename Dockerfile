FROM python:3.13-slim

ARG DEBIAN_MIRROR=http://mirrors.aliyun.com/debian
ARG DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security
ARG PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_INDEX_URL=${PYPI_INDEX_URL} \
    PATH="/opt/venv/bin:${PATH}" \
    APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

RUN sed -i \
      -e "s|http://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
      -e "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      libegl1 \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      libsm6 \
      libxt6 \
      libx11-6 \
      libxext6 \
      libxrender1 \
      unar \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -i "${PYPI_INDEX_URL}" uv

COPY pyproject.toml uv.lock .python-version ./
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt \
    && pip install --no-cache-dir -i "${PYPI_INDEX_URL}" -r requirements.txt

COPY app ./app
COPY run.py ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
