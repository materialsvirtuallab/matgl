# We start from a uv image to make life easier.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y software-properties-common gpg gpg-agent && \
    apt-get update && \
    apt-get upgrade --no-install-recommends -y && \
    apt-get install --no-install-recommends -y \
        bc \
        build-essential \
        ccache \
        clang \
        clang-format \
        cmake \
        cmake-curses-gui \
        curl \
        doxygen \
        enchant-2 \
        g++ \
        gcc \
        gfortran \
        git \
        hdf5-tools \
        less \
        libblas-dev \
        libeigen3-dev \
        libenchant-2-dev \
        libfftw3-dev \
        libgsl-dev \
        libhdf5-serial-dev \
        libhwloc-dev \
        libjpeg-dev \
        liblapack-dev \
        libnetcdf-dev \
        libomp-dev \
        libopenblas-dev \
        libnuma-dev \
        libpng-dev \
        libproj-dev \
        libreadline-dev \
        libyaml-dev \
        libzstd-dev \
        make \
        ninja-build


RUN git clone https://github.com/advancesoftcorp/lammps.git

RUN mkdir -p lammps/build-serial && \
    cd lammps/build-serial && \
    cmake -C ../cmake/presets/basic.cmake -D BUILD_SHARED_LIBS=on -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on -D PKG_ML-GNNP=on -D PKG_EXTRA-COMPUTE=on -D PKG_EXTRA-FIX=on -D PKG_MANYBODY=on -D PKG_EXTRA-DUMP=on -D PKG_MOLECULE=on ../cmake && \
    make -j 8 && \
    make install && \
    cp lmp /usr/local/bin

ENV PATH="/usr/local/bin:${PATH}"

RUN uv pip install matgl --system
