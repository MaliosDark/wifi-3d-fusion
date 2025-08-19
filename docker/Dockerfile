FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip git tcpdump libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN bash scripts/install_all.sh
CMD ["bash","scripts/run_realtime.sh","--source","esp32"]
