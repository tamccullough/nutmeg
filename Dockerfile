FROM python:3.7
ADD . /canples
WORKDIR /canples
RUN pip install -r requirements.txt
CMD gunicorn -b 0.0.0.0:5000 app:canples --daemon && \
      sed -i -e 's/$PORT/'"$PORT"'/g' /etc/nginx/conf.d/default.conf && \
      nginx -g 'daemon off;
