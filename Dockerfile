FROM python:3.7.5
WORKDIR /usr/src/tltk
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN cd robustness ; make

