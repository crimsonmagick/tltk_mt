FROM python:3.7.5
WORKDIR /usr/src/tltk
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get -y install vim nano
RUN cd robustness ; make
RUN echo 'export PYTHONPATH="/usr/src/tltk/:/usr/src/tltk/robustness:$PYTHONPATH"' >> ~/.bashrc 
RUN apt-get update

