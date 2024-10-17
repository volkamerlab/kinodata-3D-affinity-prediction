#!/bin/bash

# Check if a suffix was provided
if [ -z "$1" ]; then
  echo "Please provide a file suffix."
  exit 1
fi

# Assign the provided suffix to a variable
suffix=$1

# Find the most recent file with the provided suffix
recent_file=$(ls -t *.$suffix 2>/dev/null | head -n 1)

# Check if a file was found
if [ -z "$recent_file" ]; then
  echo "No files found with suffix .$suffix"
  exit 1
fi

# Print the contents of the most recent file
cat "$recent_file"
