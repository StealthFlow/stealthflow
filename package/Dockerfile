FROM python:3.8

ENV DEBCONF_NOWARNINGS yes
COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y --quiet --no-install-recommends
RUN pip install -q --upgrade pip
RUN pip install -r requirements.txt -q

WORKDIR /work
CMD ["/bin/bash"]