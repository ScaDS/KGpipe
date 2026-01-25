#!/bin/bash
# Will create a new experiment directory to build a pipeline with the KGpipe framework.

set -e  # Exit on error

# get directory of this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# get current directory
CURRENT_DIR=$(pwd)

# name of the experiment
read -p "Enter the name of the experiment: " EXPERIMENT_NAME

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "Error: Experiment name cannot be empty"
    exit 1
fi

# Convert to valid Python package name (lowercase, underscores, no spaces)
PACKAGE_NAME=$(echo "$EXPERIMENT_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')

# Ask for target directory
read -p "Enter target directory (default: $CURRENT_DIR): " TARGET_DIR
TARGET_DIR=${TARGET_DIR:-$CURRENT_DIR}

# Full path to new experiment directory
NEW_EXPERIMENT_DIR="$TARGET_DIR/$EXPERIMENT_NAME"

# Check if directory already exists
if [ -d "$NEW_EXPERIMENT_DIR" ]; then
    read -p "Directory $NEW_EXPERIMENT_DIR already exists. Overwrite? (y/N): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    rm -rf "$NEW_EXPERIMENT_DIR"
fi

echo "Creating new experiment at: $NEW_EXPERIMENT_DIR"
mkdir -p "$NEW_EXPERIMENT_DIR"

# Copy files, excluding logs, cache, and init.sh
echo "Copying project files..."
rsync -av --exclude='app.log' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='init.sh' \
         --exclude='.git' \
         "$SCRIPT_DIR/" "$NEW_EXPERIMENT_DIR/"

# Create src directory structure
mkdir -p "$NEW_EXPERIMENT_DIR/src/$PACKAGE_NAME"

# Copy and rename package directory
if [ -d "$SCRIPT_DIR/src/kgpipe_examples" ]; then
    cp -r "$SCRIPT_DIR/src/kgpipe_examples"/* "$NEW_EXPERIMENT_DIR/src/$PACKAGE_NAME/"
    # Remove the old directory if it was copied
    rm -rf "$NEW_EXPERIMENT_DIR/src/kgpipe_examples" 2>/dev/null || true
fi

# Copy docker directory if it exists
if [ -d "$SCRIPT_DIR/src/docker" ]; then
    cp -r "$SCRIPT_DIR/src/docker" "$NEW_EXPERIMENT_DIR/src/"
fi

# Update pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/name = \"kgpipe-examples\"/name = \"kgpipe-$PACKAGE_NAME\"/" "$NEW_EXPERIMENT_DIR/pyproject.toml"
sed -i "s/description = \"Examples for KGpipe\"/description = \"$EXPERIMENT_NAME project for KGpipe\"/" "$NEW_EXPERIMENT_DIR/pyproject.toml"

# Replace package name in Python files
echo "Updating package references in Python files..."
find "$NEW_EXPERIMENT_DIR/src/$PACKAGE_NAME" -type f -name "*.py" -exec sed -i "s/kgpipe_examples/$PACKAGE_NAME/g" {} +

echo ""
echo "✓ Experiment '$EXPERIMENT_NAME' initialized successfully!"
echo "  Location: $NEW_EXPERIMENT_DIR"
echo "  Package name: $PACKAGE_NAME"
echo ""
echo "Next steps:"
echo "  1. cd $NEW_EXPERIMENT_DIR"
echo "  2. Install dependencies: pip install -e ."
echo "  3. Start developing your pipeline!"
