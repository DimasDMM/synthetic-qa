#!/bin/bash

MANAGER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $MANAGER_DIR
case $1 in
  docker:run)
    ${MANAGER_DIR}/run-docker.sh "$2" "$3"
    ;;
  docker:down)
    ${MANAGER_DIR}/down-docker.sh
    ;;
  *)
    echo "Error: The command does not exist!!"
    exit 1
    ;;
esac
