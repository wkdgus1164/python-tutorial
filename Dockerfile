FROM python:3.10-slim

# slim alpine buster stretch jessie wheezy

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["flask", "--app", "api", "run", "--host=0.0.0.0"]