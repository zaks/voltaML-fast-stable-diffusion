FROM stax124/aitemplate:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install curl -y
RUN curl -sL https://deb.nodesource.com/setup_18.x | bash

RUN apt install nodejs -y

RUN npm i -g yarn
RUN apt install time git -y
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements /app/requirements

RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/api.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/bot.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/pytorch.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install -r requirements/interrogation.txt
RUN --mount=type=cache,mode=0755,target=/root/.cache/pip pip install python-dotenv

COPY . /app

RUN --mount=type=cache,mode=0755,target=/app/frontend/node_modules cd frontend && yarn install && yarn build
RUN rm -rf frontend/node_modules

# Install ssh
RUN apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:rootme123' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config && \
    echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

# Fix cache linkage for runpod
RUN rm -rf /root/.cache /app/data
RUN ln -s /workspace/cache /root/.cache && ln -s /workspace/app-data /app/data


# Run the server
RUN chmod +x scripts/start.sh
ENTRYPOINT ["bash", "./scripts/start.sh"]
CMD [ "/usr/sbin/sshd" ]