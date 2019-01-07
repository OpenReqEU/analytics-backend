# Use an official Python runtime as a parent image
FROM python:2.7

ENV LANG             it
ENV SPACY_VERSION    2.0.11

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

########################################
# hdbscan
########################################
RUN pip install hdbscan==0.8.12

########################################
# spaCy
########################################
RUN pip install spacy==2.0.11
RUN pip install scikit-learn==0.19.1
RUN python -m spacy download ${LANG}
RUN python -m spacy download en_core_web_sm

# Install any needed packages specified in requirements.txt
RUN pip2 install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

RUN python -m nltk.downloader punkt

RUN python -m nltk.downloader stopwords

# Run app.py when the container launches
CMD ["python", "app.py"]