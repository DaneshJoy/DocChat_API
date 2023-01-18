FROM tiangolo/uvicorn-gunicorn-fastapi-docker

WORKDIR /app

# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE 1

# ensures that the python output is sent straight to terminal (e.g. your container log)
# without being first buffered and that you can see the output of your application (e.g. django logs)
# in real time. Equivalent to python -u: https://docs.python.org/3/using/cmdline.html#cmdoption-u
ENV PYTHONUNBUFFERED 1
ENV ENVIRONMENT prod
ENV TESTING 0

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/
RUN chmod +x run.sh

ENV PYTHONPATH=/app

# chown all the files to the app user
RUN chown -R app:app $HOME

CMD ["./run.sh"]

# CMD ["uvicorn", "docchat_app:app", "--host", "0.0.0.0", "--port", "80"]
# CMD ["gunicorn" "-k" "uvicorn.workers.UvicornWorker" d"occhat_app:app"]
