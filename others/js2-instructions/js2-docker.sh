#!/bin/bash
GROUP=$1
PORT=$2
docker run --rm --network host -p $PORT:5800 -it js2client $GROUP $PORT
