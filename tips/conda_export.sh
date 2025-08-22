#!/bin/bash
#
# The script export installed packages to requireements.txt
#
# Usage: conda_export path_to/requireements.txt
# location: ~/.local/bin/conda_export
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-18
# * CREATED ON: 2025-08-18
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh-USA. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh-USA and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh-USA
# -----------------------------------------------------------------------------
# Set conda path directly based on your installation
CONDA_BASE="$HOME/anaconda3"
export PATH="$CONDA_BASE/bin:$PATH"

# Source conda initialization
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

# Verify conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command not found at $CONDA_BASE/bin/conda"
    exit 1
fi

# Get output filename from argument or use default
OUTPUT_FILE=${1:-"environment_simplified.yml"}

# Create temporary files
TMP_ENV_EXPORT=$(mktemp)
TMP_CONDA_LIST=$(mktemp)
TMP_OUTPUT=$(mktemp)
TMP_PIP=$(mktemp)

# Capture conda environment export and list
conda env export > "$TMP_ENV_EXPORT" 2>/dev/null
conda list > "$TMP_CONDA_LIST"

# Get current environment name
CURRENT_ENV=$(grep "^name:" "$TMP_ENV_EXPORT" | cut -d' ' -f2)

# Start building output file
echo "name: $CURRENT_ENV" > "$TMP_OUTPUT"

# Extract channels from env export
echo "channels:" >> "$TMP_OUTPUT"
sed -n '/^channels:/,/^dependencies:/p' "$TMP_ENV_EXPORT" | grep "^[[:space:]]*-" | grep -v "dependencies:" | sed 's/^/  /' >> "$TMP_OUTPUT"

# Add dependencies header
echo "dependencies:" >> "$TMP_OUTPUT"

# Add conda packages (excluding pip and packages beginning with _)
tail -n +4 "$TMP_CONDA_LIST" | grep -v "<pip>" | while read -r LINE; do
    PKG_NAME=$(echo "$LINE" | awk '{print $1}')
    PKG_VERSION=$(echo "$LINE" | awk '{print $2}')

    # Skip empty lines and packages that begin with underscore
    if [[ -z "$PKG_NAME" ]] || [[ "$PKG_NAME" == _* ]]; then
        continue
    fi

    echo "  - $PKG_NAME=$PKG_VERSION" >> "$TMP_OUTPUT"
done

# Try to extract pip packages from environment.yml with filtering
if grep -q "^[[:space:]]*- pip:" "$TMP_ENV_EXPORT"; then
    echo "  - pip:" >> "$TMP_OUTPUT"

    # Extract pip packages and filter out those starting with underscore
    sed -n '/^[[:space:]]*- pip:/,/^[[:space:]]*prefix:/p' "$TMP_ENV_EXPORT" | \
    grep -v "^[[:space:]]*- pip:" | \
    grep -v "^[[:space:]]*prefix:" | \
    sed 's/^.*-[[:space:]]*//' > "$TMP_PIP"

    # Filter out packages starting with underscore
    while read -r pip_line; do
        if [[ -n "$pip_line" ]]; then
            # Extract package name (everything before == or = or space)
            pkg_name=$(echo "$pip_line" | sed 's/==.*//' | sed 's/=.*//' | awk '{print $1}')
            # Only add if doesn't start with underscore
            if [[ "$pkg_name" != _* ]]; then
                echo "    - $pip_line" >> "$TMP_OUTPUT"
            fi
        fi
    done < "$TMP_PIP"
else
    # Fallback: check conda list for pip packages
    if grep -q "<pip>" "$TMP_CONDA_LIST"; then
        echo "  - pip:" >> "$TMP_OUTPUT"
        grep "<pip>" "$TMP_CONDA_LIST" | while read -r LINE; do
            PKG_NAME=$(echo "$LINE" | awk '{print $1}')
            PKG_VERSION=$(echo "$LINE" | awk '{print $2}')
            # Don't add pip packages that start with underscore either
            if [[ "$PKG_NAME" != _* ]]; then
                echo "    - $PKG_NAME==$PKG_VERSION" >> "$TMP_OUTPUT"
            fi
        done
    fi
fi

# Move temp output to final destination
mv "$TMP_OUTPUT" "$OUTPUT_FILE"

# Clean up temp files
rm -f "$TMP_ENV_EXPORT" "$TMP_CONDA_LIST" "$TMP_PIP"

echo "Environment exported to $OUTPUT_FILE"