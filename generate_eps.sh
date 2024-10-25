#!/bin/bash

# Check if a PlantUML file was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <plantuml_file>"
  exit 1
fi

PLANTUML_FILE="$1"
OUTPUT_FILE="${PLANTUML_FILE%.*}.eps"

# Install PlantUML and required dependencies
sudo apt-get update
sudo apt-get install -y default-jre graphviz

# Generate EPS from the PlantUML file
plantuml -teps "$PLANTUML_FILE"

# Check if the EPS file was created
if [ -f "$OUTPUT_FILE" ]; then
  echo "EPS file generated successfully: $OUTPUT_FILE"
else
  echo "Failed to generate EPS file."
  exit 1
fi
