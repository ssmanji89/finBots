FROM python:3.9-slim

RUN apt-get update -y --fix-missing \
    && apt-get upgrade -y \
    && apt-get install -y git pip tzdata
RUN apt-get update 

# Set the timezone
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
COPY . /botStuff/   

WORKDIR /botStuff
COPY botstuff.json /botStuff/botstuff.json
RUN pip install --no-cache-dir --no-input --disable-pip-version-check -r requirements.txt 

ARG curUser
ARG curPass
ARG curTotp
ENTRYPOINT ["python3", "/botStuff/_.stocks/_regAnalysis_stocks.py"] 

# docker build --no-cache -t stocks_botstuff_org .; docker run -d -e curUser='ssman@gmail.com' -e curPass='!!@#%^*' -e curTotp='V' stocks_botstuff_org