From python:3.0
WORKDIR /opt/python 
RUN apt update && apt install -y cmake gcc g++ python3-opencv
RUN pip3 install -i https://pypi.douban.com/simple/ dlib
RUN apt autoclean && apt clean
RUN rm -rf /tmp/*
CMD ["python3", "apply.py"]
