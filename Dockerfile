FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY requirements.txt .

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -U -r requirements.txt


FROM debian:bookworm-slim AS runtime

# Добавлен пакет openssl
RUN apt-get update && \
    apt-get install -y curl ffmpeg python3 openssl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /Whisper-WebUI/venv /Whisper-WebUI/venv

RUN openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

RUN sed -i '76s/weights_only=weights_only,/weights_only=False,/' /Whisper-WebUI/venv/lib/python3.11/site-packages/lightning_fabric/utilities/cloud_io.py

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

ENV PATH="/Whisper-WebUI/venv/bin:$PATH"
ENV LD_LIBRARY_PATH=/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cublas/lib:/Whisper-WebUI/venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

ENTRYPOINT [ "python", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--ssl_certfile", "cert.pem", "--ssl_keyfile", "key.pem", "--ssl_verify", "False"]
