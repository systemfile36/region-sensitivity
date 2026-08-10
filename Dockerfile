FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN bash /tmp/install_deps.sh

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pyproject.toml README.md /app/
COPY ssat /app/ssat
RUN pip install --no-cache-dir --no-deps /app

ENTRYPOINT ["ssat"]
CMD ["--help"]
