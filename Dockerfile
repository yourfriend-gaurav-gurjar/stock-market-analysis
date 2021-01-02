# Dockerfile for tensorflow 2 with gpu
FROM rapidsai/rapidsai

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y opensjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java &&
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME --useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-opensjdk-amd64/
RUN export JAVA_HOME

# build your custom image from using the following command
dockre build -t gauravgurjar .
