FROM python:3.9.17

RUN pip install matgl jupyter lxml
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
RUN pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

ENV DGLBACKEND=pytorch

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--no-browser"]
