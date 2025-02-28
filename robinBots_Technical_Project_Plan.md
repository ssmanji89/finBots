
# Technical Project Plan for the robinBots Project

## 1. Project Overview
The robinBots project involves creating a programmatic trading system that iteratively works through users' open positions and executes suggested trades based on a market-determined schedule. This project will be developed using Python, containerized with Docker, and managed through a GitHub repository. The entire development process will be carried out using VS Code on a MacOS M1, with GitHub Desktop and Bash for version control and repository management.

## 2. Project Milestones
1. Repository Setup and Configuration
2. Initial Python Application Development
3. Dockerization of the Python App
4. Stripe Integration for Payment Processing
5. Deployment to DigitalOcean
6. Testing and Quality Assurance
7. Documentation and Final Review

## Milestone 1: Repository Setup and Configuration

**Objective:**
Create and configure a GitHub repository to manage the source code, track issues, and facilitate collaboration.

**Steps:**

1. **Create a GitHub Repository:**
   - **Via GitHub Desktop:**
     - Open GitHub Desktop on your MacOS M1.
     - Click on `File` > `New Repository`.
     - Name the repository `robinBots`.
     - Set the local path where you want to store the repository on your Mac.
     - Choose `Python` as the `.gitignore` template.
     - Initialize the repository with a `README.md`.
     - Commit the initial files and publish the repository to GitHub.

   - **Via Bash:**
     - Open the terminal on your Mac.
     - Navigate to the desired directory:
       ```bash
       cd ~/Projects
       ```
     - Initialize a new GitHub repository:
       ```bash
       mkdir robinBots
       cd robinBots
       git init
       echo "# robinBots" >> README.md
       git add .
       git commit -m "Initial commit"
       git branch -M main
       git remote add origin https://github.com/yourusername/robinBots.git
       git push -u origin main
       ```

2. **Setup Repository Structure:**
   - Create the necessary folders and files for your project:
     ```bash
     mkdir src
     mkdir tests
     mkdir docker
     touch src/main.py
     touch requirements.txt
     touch docker/Dockerfile
     ```
   - Update the `README.md` to include a brief overview of the project.

3. **Configure GitHub Actions for CI/CD:**
   - Create a `.github/workflows` directory:
     ```bash
     mkdir -p .github/workflows
     ```
   - Create a `ci.yml` file to automate testing and deployment:
     ```yaml
     name: Python application

     on: [push]

     jobs:
       build:

         runs-on: ubuntu-latest

         steps:
         - uses: actions/checkout@v2
         - name: Set up Python
           uses: actions/setup-python@v2
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: |
             python -m pip install --upgrade pip
             pip install -r requirements.txt
         - name: Run tests
           run: |
             pytest
     ```

## Milestone 2: Initial Python Application Development

**Objective:**
Develop the core functionality of the `robinBots` trading system.

**Steps:**

1. **Set Up the Development Environment in VS Code:**
   - Open the `robinBots` repository in VS Code.
   - Ensure that Python is installed on your MacOS M1 and that VS Code is set up with the Python extension.
   - Create a virtual environment for the project:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Install necessary packages and update `requirements.txt`:
     ```bash
     pip install robin_stocks pandas numpy
     pip freeze > requirements.txt
     ```

2. **Develop the Core Trading Logic:**
   - Implement the initial version of `main.py`:
     ```python
     import robin_stocks.robinhood as r
     import pandas as pd

     def login():
         r.login(username='your_username', password='your_password')

     def get_open_positions():
         return r.account.get_open_stock_positions()

     def execute_trades():
         positions = get_open_positions()
         # Logic to execute trades based on market conditions

     if __name__ == "__main__":
         login()
         execute_trades()
     ```

3. **Testing:**
   - Create unit tests in the `tests/` directory using `pytest` to validate the functionality of your trading logic.
   - Run the tests:
     ```bash
     pytest tests/
     ```

4. **Version Control:**
   - Commit the initial development progress:
     ```bash
     git add .
     git commit -m "Initial trading logic implemented"
     git push
     ```

## Milestone 3: Dockerization of the Python App

**Objective:**
Containerize the `robinBots` application using Docker to ensure consistent execution across different environments.

**Steps:**

1. **Create a Dockerfile:**
   - Implement the Dockerfile in the `docker/` directory:
     ```Dockerfile
     FROM python:3.9-slim

     WORKDIR /app

     COPY requirements.txt requirements.txt
     RUN pip install --no-cache-dir -r requirements.txt

     COPY . .

     CMD ["python", "src/main.py"]
     ```

2. **Build and Test the Docker Image:**
   - Build the Docker image:
     ```bash
     docker build -t robinbots:latest .
     ```
   - Run the Docker container to ensure everything works:
     ```bash
     docker run --rm robinbots:latest
     ```

3. **Version Control:**
   - Commit the Dockerfile and any related changes:
     ```bash
     git add docker/Dockerfile
     git commit -m "Added Docker support"
     git push
     ```

## Milestone 4: Stripe Integration for Payment Processing

**Objective:**
Integrate Stripe into the application for handling subscriptions and payments.

**Steps:**

1. **Set Up Stripe Account and API Keys:**
   - Sign up for Stripe and obtain your API keys.

2. **Implement Stripe in the Application:**
   - Add Stripe to your project by updating `requirements.txt` and installing:
     ```bash
     pip install stripe
     pip freeze > requirements.txt
     ```
   - Update `main.py` to include subscription management:
     ```python
     import stripe

     stripe.api_key = "your_secret_key"

     def create_checkout_session():
         session = stripe.checkout.Session.create(
             payment_method_types=['card'],
             line_items=[{
                 'price_data': {
                     'currency': 'usd',
                     'product_data': {
                         'name': 'Subscription to robinBots',
                     },
                     'unit_amount': 999,
                 },
                 'quantity': 1,
             }],
             mode='subscription',
             success_url='https://your_domain/success',
             cancel_url='https://your_domain/cancel',
         )
         return session.url
     ```

3. **Webhook Integration:**
   - Set up a webhook to handle Stripe events, such as subscription creation or failure.

4. **Testing:**
   - Test the Stripe integration using the test keys and environment.

5. **Version Control:**
   - Commit the changes related to Stripe integration:
     ```bash
     git add .
     git commit -m "Added Stripe integration"
     git push
     ```

## Milestone 5: Deployment to DigitalOcean

**Objective:**
Deploy the Dockerized application to a DigitalOcean Droplet for production use.

**Steps:**

1. **Set Up the DigitalOcean Environment:**
   - SSH into your existing Debian Droplet.
   - Pull the latest code from the GitHub repository:
     ```bash
     git clone https://github.com/yourusername/robinBots.git
     cd robinBots
     ```

2. **Deploy the Dockerized Application:**
   - Build and run the Docker container on the Droplet:
     ```bash
     docker build -t robinbots:latest .
     docker run -d --name robinbots -p 80:80 robinbots:latest
     ```

3. **Nginx Configuration:**
   - If not already configured, set up Nginx to reverse proxy to the Docker container.

4. **Monitor the Deployment:**
   - Ensure the application is running correctly and that Stripe payments are processed as expected.

5. **Version Control:**
   - Ensure all deployment-related scripts and configurations are committed:
     ```bash
     git add .
     git commit -m "Deployment scripts and configs"
     git push
     ```

## Milestone 6: Testing and Quality Assurance

**Objective:**
Perform thorough testing to ensure the system works as expected.

**Steps:**

1. **System Testing:**
   - Test the entire system, from user subscription to trade execution, in a controlled environment.

2. **Load Testing:**
   - Simulate high traffic to ensure the system can handle multiple users concurrently.

3. **Security Testing:**
   - Conduct a security audit to ensure the application and payment processing are secure.

4. **Bug Fixes:**
   - Address any issues discovered during testing.

5. **Version Control:**
   - Commit all changes after testing and bug fixes:
     ```bash
     git add .
     git commit -m "Final testing and bug fixes"
     git push
     ```
