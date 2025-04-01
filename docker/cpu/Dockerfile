FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ENV DGLBACKEND=pytorch

RUN uv pip install matgl jupyter lxml

CMD [ "python", "-c", "import matgl; print(matgl.__version__)"]
