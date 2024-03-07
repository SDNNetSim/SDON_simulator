#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: make_venv.sh <target_directory> <python_version>"
  exit 1
fi

if ! command -v "$2" &> /dev/null; then
  echo "Python version '$2' could not be found. Please install it or choose a valid version."
  exit 1
fi

cd "$1" || { echo "Directory '$1' does not exist."; exit 1; }

"$2" -m venv venv

echo "Virtual environment 'venv' created in '$1' using $2!"
