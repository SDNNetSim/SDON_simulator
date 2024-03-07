#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: register_rl_env.sh <algorithm> <env_name>"
  exit 1
fi

python ai_scripts/register_env.py --algo "$1" --env-name "$2"
