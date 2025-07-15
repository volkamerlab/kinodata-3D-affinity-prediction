#!/bin/bash

# Script to copy epoch*.ckpt files to best.ckpt
# Usage: ./rename_best_checkpoint.sh [-R] [directory]

show_usage() {
    echo "Usage: $0 [-R] [directory]"
    echo "  -R        Apply recursively to all subdirectories"
    echo "  directory Directory to process (default: current directory)"
    echo ""
    echo "This script copies epoch*.ckpt files to best.ckpt only if exactly one such file exists."
}

process_directory() {
    local dir="$1"
    local found_files=()
    
    # Find all epoch*.ckpt files in the directory
    while IFS= read -r -d '' file; do
        found_files+=("$file")
    done < <(find "$dir" -maxdepth 1 -name "epoch*.ckpt" -type f -print0 2>/dev/null)
    
    # Check if exactly one epoch*.ckpt file exists
    if [ ${#found_files[@]} -eq 1 ]; then
        local source_file="${found_files[0]}"
        local target_file="$dir/best.ckpt"
        
        # Check if best.ckpt already exists
        if [ -f "$target_file" ]; then
            echo "Warning: $target_file already exists in $dir, skipping..."
            return 1
        fi
        
        # Copy the file
        if cp "$source_file" "$target_file"; then
            echo "Copied: $(basename "$source_file") -> best.ckpt in $dir"
            return 0
        else
            echo "Error: Failed to copy $source_file to $target_file"
            return 1
        fi
    elif [ ${#found_files[@]} -eq 0 ]; then
        echo "No epoch*.ckpt files found in $dir"
        return 0
    else
        echo "Multiple epoch*.ckpt files found in $dir (${#found_files[@]} files), skipping..."
        for file in "${found_files[@]}"; do
            echo "  - $(basename "$file")"
        done
        return 1
    fi
}

# Parse command line arguments
recursive=false
target_dir="."

while [[ $# -gt 0 ]]; do
    case $1 in
        -R)
            recursive=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            target_dir="$1"
            shift
            ;;
    esac
done

# Validate target directory
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist"
    exit 1
fi

# Make target_dir absolute for consistency
target_dir=$(cd "$target_dir" && pwd)

echo "Processing directory: $target_dir"
if [ "$recursive" = true ]; then
    echo "Mode: Recursive"
else
    echo "Mode: Single directory"
fi
echo ""

# Process directories
if [ "$recursive" = true ]; then
    # Process all subdirectories recursively
    processed=0
    successful=0
    
    while IFS= read -r -d '' dir; do
        ((processed++))
        if process_directory "$dir"; then
            ((successful++))
        fi
    done < <(find "$target_dir" -type d -print0)
    
    echo ""
    echo "Summary: Processed $processed directories, successfully copied files in $successful directories"
else
    # Process only the target directory
    process_directory "$target_dir"
fi
