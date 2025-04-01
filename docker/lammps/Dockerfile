# We start from a uv image to make life easier.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update -y && apt-get install -y \
    build-essential \
    git \
    cmake \
    g++ \
    gcc \
    gfortran \
    python3 \
    python3-dev

RUN git clone https://github.com/advancesoftcorp/lammps.git

RUN mkdir -p lammps/build-serial && \
    cd lammps/build-serial && \
    cmake -C ../cmake/presets/basic.cmake -D BUILD_SHARED_LIBS=on -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D PKG_ML-GNNP=on -D PKG_EXTRA-COMPUTE=on -D PKG_EXTRA-FIX=on -D PKG_MANYBODY=on -D PKG_EXTRA-DUMP=on -D PKG_MOLECULE=on ../cmake && \
    make -j 8 && \
    make install && \
    cp lmp /usr/local/bin

ENV PATH="/usr/local/bin:${PATH}"
ENV DGLBACKEND=pytorch
ENV PYTHONPATH="/lammps/src/ML-GNNP:${PYTHONPATH}"

RUN uv pip install matgl --system
