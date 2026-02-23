#!/usr/bin/env bash
set -euo pipefail

validate_file() {
  local filename="$1"

  # Check if file matches the expected pattern: NNNN-*.md
  if ! echo "$filename" | grep -qE '^[0-9]{4}-[a-zA-Z0-9_-]+\.md$'; then
    echo "ERROR: Invalid filename: $filename"
    echo "  Expected format: NNNN-name.md (e.g., 0007-my-proposal.md)"
    return 1
  fi

  return 0
}

exit_code=0
valid_ids=""

# Find all files in proposals/ directory (if it exists)
if [ -d "proposals" ]; then
  for file in $(ls proposals/ | sort); do
    file="proposals/$file"
    # Skip if no files match (glob returns literal pattern)
    [ -e "$file" ] || continue

    filename=$(basename "$file")

    if validate_file "$filename"; then
      echo "OK: $filename"
      # Collect ID and filename for duplicate checking
      id="${filename:0:4}"
      valid_ids="${valid_ids}${id} ${filename}"$'\n'
    else
      exit_code=1
    fi
  done

  # Check for duplicate IDs
  duplicate_ids=$(echo "$valid_ids" | awk '{print $1}' | sort | uniq -d)
  if [ -n "$duplicate_ids" ]; then
    for dup_id in $duplicate_ids; do
      files=$(echo "$valid_ids" | grep "^$dup_id " | awk '{print $2}' | tr '\n' ' ')
      echo "ERROR: Duplicate ID $dup_id found in: $files"
    done
    exit_code=1
  fi
fi

exit $exit_code
