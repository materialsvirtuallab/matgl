FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ENV DGLBACKEND=pytorch

RUN uv pip install matgl jupyter lxml

CMD ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0", "--no-browser"]
