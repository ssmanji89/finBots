FROM python:3.9-slim

ENV TZ="America/Chicago"
RUN apt-get update -y --fix-missing \
    && apt-get upgrade -y \
    && apt-get install -y git tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /botStuff

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /botStuff/_CryptoBots/cryptoBot_requirements.txt \
    && pip install --no-cache-dir -r /botStuff/_StockBot/stockBot_requirements.txt 

CMD ["sh", "-c", "python3 /botStuff/_CryptoBots/cryptoBot.py & python3 /botStuff/_StockBot/stockBot.py"]
