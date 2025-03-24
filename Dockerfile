# set base image (host OS)
FROM python:3.11

# set the working directory in the container
WORKDIR /code

# copy the content of the local directory to the working directory
COPY . /code

# install dependencies
RUN pip install --upgrade pip
RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html
RUN pip install -e '.[ci]'

# command to run on container start
CMD [ "pytest", "tests" ]
