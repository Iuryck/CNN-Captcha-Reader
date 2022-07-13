FROM python:3.8.8
ENV PYTHONUNBUFFERED 1

WORKDIR /NoCaptcha
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/Iuryck/CNN-Captcha-Reader.git 
COPY . .
