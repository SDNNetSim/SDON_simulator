#!/bin/bash

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <username> <partition>"
  exit 1
fi

USER_NAME=$1
PARTITION=$2

USER_PRIORITY=$(squeue -p $PARTITION -t PD --noheader --format="%u %Q" | grep "^$USER_NAME " | awk '{print $2}' | sort -nr | head -n 1)

if [[ ! -z $USER_PRIORITY ]]; then
  HIGHER_PRIORITY_JOBS=$(squeue -p $PARTITION -t PD --noheader --format="%Q" | awk -v up="$USER_PRIORITY" '$1 > up' | wc -l)
fi

HIGHEST_PRIORITY=$(squeue -p $PARTITION -t PD --noheader --format="%Q" | sort -nr | head -n 1)

if [[ -z $USER_PRIORITY ]]; then
  echo "User: $USER_NAME has no pending jobs in the '$PARTITION' partition."
else
  echo "User: $USER_NAME has priority $USER_PRIORITY in the '$PARTITION' partition, and the highest priority in the queue is currently $HIGHEST_PRIORITY."
  echo "$HIGHER_PRIORITY_JOBS jobs have a higher priority than $USER_NAME's highest priority job in the '$PARTITION' partition."
fi
