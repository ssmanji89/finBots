FROM python:3.9-slim

# Set the timezone
ENV TZ="America/Chicago"

# Install curl and unzip
RUN apt-get update \
    && apt-get install -y curl oathtool unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and unzip the Bitwarden CLI
RUN curl -LsS -o /tmp/bw-linux.zip https://github.com/bitwarden/cli/releases/download/v1.22.1/bw-linux-1.22.1.zip \
    && unzip /tmp/bw-linux.zip -d / \
    && chmod +x /usr/local/bin/bw \
    && rm /tmp/bw-linux.zip

WORKDIR /botStuff

# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r /botStuff/_CryptoBots/cryptoBot_requirements.txt \
    && pip install -r /botStuff/_StockBot/stockBot_requirements.txt

# Create the fetch_credentials.sh script
RUN echo "#!/bin/bash" > /botStuff/fetch_credentials.sh
RUN echo "BW_USERNAME=\$BW_USERNAME" >> /botStuff/fetch_credentials.sh
RUN echo "BW_PASSWORD=\$BW_PASSWORD" >> /botStuff/fetch_credentials.sh
RUN echo "BW_TOTP_CODE=\$(oathtool --totp -b \$BW_TOTP_SECRET)" >> /botStuff/fetch_credentials.sh
RUN echo "bw login \$BW_USERNAME \$BW_PASSWORD --method 0 --code \$BW_TOTP_CODE" >> /botStuff/fetch_credentials.sh
RUN echo "SECRET_USERNAME=\$(bw get username user.robinhood.com_)" >> /botStuff/fetch_credentials.sh
RUN echo "SECRET_PASSWORD=\$(bw get password user.robinhood.com_)" >> /botStuff/fetch_credentials.sh
# Make the script executable
RUN chmod +x /botStuff/fetch_credentials.sh

# Source the fetch_credentials.sh script to set environment variables
# Make sure the fetch_credentials.sh script is executable
RUN chmod +x /botStuff/fetch_credentials.sh
RUN /bin/bash -c "/botStuff/fetch_credentials.sh"

# Command to run the bots
CMD ["sh", "-c", "python3 /botStuff/_CryptoBots/cryptoBot.py & python3 /botStuff/_StockBot/stockBot.py"]

