FROM python:3.11

ENV DGLBACKEND=pytorch

RUN pip install --upgrade pip
RUN pip install matgl jupyter lxml

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--no-browser"]
