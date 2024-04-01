#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: register_rl_env.sh <algo> <env_name>"
  exit 1
fi

python ai_scripts/register_env.py --algo "$1" --env-name "$2"
cat >> venvs/unity_venv/venv/lib/python3.11/site-packages/rl_zoo3/import_envs.py <<EOL

from run_ai_sim import SimEnv

register(id='SimEnv', entry_point='run_ai_sim:SimEnv')
EOL

