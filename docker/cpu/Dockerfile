FROM python:3.9.17

RUN pip install matgl jupyter lxml

ENV DGLBACKEND=pytorch

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--no-browser"]
