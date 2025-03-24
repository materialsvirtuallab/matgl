# set base image (host OS)
FROM python:3.11

# set the working directory in the container
WORKDIR /code

ENV DGLBACKEND=pytorch

# Copy required files to the workdir.
COPY src /code/src
COPY pyproject.toml /code/pyproject.toml
COPY tests /code/tests
COPY run_tests.sh /code/run_tests.sh

# install dependencies
RUN pip install --upgrade pip
RUN pip install -e '.[ci]'

# command to run on container start
CMD [ "./run_tests.sh" ]
