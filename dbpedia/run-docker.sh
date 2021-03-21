#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPT_PATH}/environment-vars.sh

if [ $# != 2 ]
then
  echo "Usage: $0 <LANG_CODE> <PORT>"
  exit
fi

# Variable with language code and external port
export LANG=$1
export PORT=$2

# Reset previous container
$SCRIPT_PATH/down-docker.sh ${LANG} ${PORT}

echo "Running docker:"
echo "- Language code: ${LANG}"
echo "- Port: ${PORT}"

cd $SCRIPT_PATH
docker-compose -f docker-compose.yml -p $PROJECT_USER up -d --build
cd -
