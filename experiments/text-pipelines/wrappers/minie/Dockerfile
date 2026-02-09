FROM maven:3.6.3-jdk-8

WORKDIR /app

RUN git clone https://github.com/uma-pi1/minie.git

WORKDIR /app/minie

COPY bin/MinIEApp.java src/main/java/MinIEApp.java

RUN mvn clean install dependency:copy-dependencies -DskipTests

COPY minie.sh /usr/local/bin/minie.sh
RUN chmod +x /usr/local/bin/minie.sh

