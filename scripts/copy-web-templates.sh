#!/bin/bash

# This script is intended to copy a custom "index.html" template to the default directory 
# sourced for running the webserver. May include other files/templates as needed.

# Add functionality to source visualization path var from the Python scripts?

VISUALIZATION_PATH="env/visualization"

if [ ! -d "$VISUALIZATION_PATH" ]; then
    mkdir -p "$VISUALIZATION_PATH"
    echo "Created: $VISUALIZATION_PATH"
else
    echo "Already exists: $VISUALIZATION_PATH"
fi

cp templates/index.html $VISUALIZATION_PATH/index.html