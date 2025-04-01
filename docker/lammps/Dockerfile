FROM ubuntu:latest

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
        libvtk9-dev \
        libyaml-dev \
        libzstd-dev \
        make \
        ninja-build


RUN git clone https://github.com/advancesoftcorp/lammps.git

RUN mkdir -p lammps/build-serial && \
    cd lammps/build-serial && \
    cmake ../cmake && \
    cmake --build . && \
    make -j 8 && \
    make install

RUN pip install matgl

CMD ["lmp"]
