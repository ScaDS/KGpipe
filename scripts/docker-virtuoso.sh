docker run \
    --name kgpipe_virtdb \
    --interactive \
    --tty \
    --env DBA_PASSWORD=mysecret \
    --publish 1111:1111 \
    --publish  8890:8890 \
    openlink/virtuoso-opensource-7:latest

#    --volume `pwd`:/database \
