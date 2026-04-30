FROM debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y curl git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    mkdir -p /Whisper-WebUI

WORKDIR /Whisper-WebUI

COPY requirements.txt .

RUN pip3 install -U -r requirements.txt --timeout 120


FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y curl ffmpeg python3 openssl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /Whisper-WebUI

COPY . .
COPY --from=builder /usr/local /usr/local

RUN openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=whisper"

RUN sed -i '76s/weights_only=weights_only,/weights_only=False,/' /usr/local/lib/python3.11/dist-packages/lightning_fabric/utilities/cloud_io.py

VOLUME [ "/Whisper-WebUI/models" ]
VOLUME [ "/Whisper-WebUI/outputs" ]

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib

ENTRYPOINT [ "python3", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860", "--ssl_certfile", "cert.pem", "--ssl_keyfile", "key.pem", "--ssl_verify", "False"]
