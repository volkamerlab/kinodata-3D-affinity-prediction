#!/bin/bash

# Usage: ./run_condor_job.sh jobfile.sub

# Ensure at least one argument is passed
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <filename>.sub"
    exit 1
fi

SUB_FILE=""
# Parse arguments and ensure .sub file is identified
for arg in "$@"; do
    if [[ "$arg" == *.sub ]]; then
        SUB_FILE="$arg"
    fi
done

if [[ -z "$SUB_FILE" ]]; then
    echo "Error: No submission file (.sub) found in arguments."
    exit 1
fi

BASE_NAME="${SUB_FILE%.sub}"

# Step 1: Git pull
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running git pull..."
git pull

# Step 2: Submit the job and capture Job ID
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitting Condor job: $SUB_FILE"
JOB_SUBMIT_OUTPUT=$(condor_submit "$SUB_FILE")
echo "$JOB_SUBMIT_OUTPUT"

JOB_ID=$(echo "$JOB_SUBMIT_OUTPUT" | awk '/submitted to cluster/ {print $6}' | tr -d '.')

if [[ -z "$JOB_ID" ]]; then
    echo "Failed to extract Job ID from submission output."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job submitted with ID: $JOB_ID"

# Step 3: Monitor job
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring job status..."
while true; do
    STATUS=$(condor_q "$JOB_ID" 2>/dev/null)
    if echo "$STATUS" | grep -q "$JOB_ID"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_ID is still running..."
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job $JOB_ID is no longer in queue (likely finished)."
        break
    fi
    sleep 5
done

# Step 4: Show output after completion
sleep 3
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetching latest log and error files..."


LATEST_LOG=$(ls -t "${BASE_NAME}"*.log 2>/dev/null | head -n1)
LATEST_OUT=$(ls -t "${BASE_NAME}"*.out 2>/dev/null | head -n1)
LATEST_ERR=$(ls -t "${BASE_NAME}"*.err 2>/dev/null | head -n1)

if [[ -f "$LATEST_LOG" ]]; then
    echo -e "\n--- Log File: $LATEST_LOG ---"
    cat "$LATEST_LOG"
else
    echo "No log file found."
fi

if [[ -f "$LATEST_OUT" ]]; then
    echo -e "\n--- Out File: $LATEST_OUT ---"
    cat "$LATEST_OUT"
else
    echo "No out file found."
fi

if [[ -f "$LATEST_ERR" ]]; then
    echo -e "\n--- Error File: $LATEST_ERR ---"
    cat "$LATEST_ERR"
else
    echo "No error file found."
fi

