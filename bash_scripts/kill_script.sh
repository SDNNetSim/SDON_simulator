#!/bin/bash

if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <script_path>"
        exit 1
fi

script_path="$1"
script_name=$(basename "$script_path")

if [ ! -f "$script_path" ]; then
        echo "Error: $script_path is not a valid file path."
        exit 1
fi

echo "$script_name has been killed."
pkill -f "$script_name"
