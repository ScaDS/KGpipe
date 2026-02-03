FROM maven:3.6.3-jdk-8

WORKDIR /app

RUN git clone https://github.com/Lambda-3/Graphene.git

WORKDIR /app/Graphene

RUN /bin/bash install-DiscourseSimplification.sh
RUN mvn -P cli clean install -DskipTests

COPY graphene.sh /usr/local/bin/graphene.sh
RUN chmod +x /usr/local/bin/graphene.sh


