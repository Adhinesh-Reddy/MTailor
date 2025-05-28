FROM debian:bookworm-slim

WORKDIR /app

COPY . .

# Install Python & pip
RUN apt update && \
    apt install -y python3 python3-pip && \
    apt clean

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
