#!/bin/bash
# Copyright 2025 Christopher Rura

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
