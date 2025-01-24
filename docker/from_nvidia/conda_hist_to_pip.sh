#!/bin/bash

# Read from standard input and process the content
while IFS= read -r line; do
    # Skip unnecessary lines
    if [[ "$line" =~ ^(name|channels|prefix): ]]; then
        continue
    fi

    # Extract dependencies
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]+([a-zA-Z0-9_-]+)(=([0-9.]+))? ]]; then
        package="${BASH_REMATCH[1]}"
        version="${BASH_REMATCH[3]}"
        if [ -z "$version" ]; then
            echo "$package"
        else
            echo "$package==$version"
        fi
    fi
done

