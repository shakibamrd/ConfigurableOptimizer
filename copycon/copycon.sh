#!/bin/bash
set -e

# --- Configuration ---

# Determine the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/copycon.cfg"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Source the config file (it should define a variable named 'source_dir')
source "$CONFIG_FILE"

if [ -z "$source_dir" ]; then
    echo "Error: 'source_dir' is not defined in $CONFIG_FILE"
    exit 1
fi

# Ensure that the src/confopt folder exists within the source directory.
SRC_FOLDER="${source_dir}/src/confopt"
if [ ! -d "$SRC_FOLDER" ]; then
    echo "Error: '$SRC_FOLDER' does not exist within the source directory."
    exit 1
fi

# --- Process command line arguments ---

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <destination_directory>"
    exit 1
fi

# Get the destination directory from the command-line argument.
DEST_DIR="$1"

# If the provided path is relative, prepend the script's directory.
if [[ "$DEST_DIR" != /* ]]; then
    DEST_DIR="$(pwd)/$DEST_DIR"
fi

# Define the target directory as a subdirectory 'confopt' in the destination directory.
TARGET_DIR="${DEST_DIR}/confopt"

# If the target directory already exists, ask for confirmation to overwrite it.
if [ -d "$TARGET_DIR" ]; then
    read -p "Directory '$DEST_DIR' already exists. Do you want to overwrite it? (y/n): " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        rm -rfv "$TARGET_DIR"
        rm -rfv "$DEST_DIR/info"
		echo ""
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

mkdir -p "$DEST_DIR"

# Create the target directory.
mkdir -p "$TARGET_DIR"

# --- Ensure source_dir is a Git repository and has no local changes ---

if [ ! -d "${source_dir}/.git" ]; then
    echo "Error: '$source_dir' is not a Git repository ('.git' folder not found)."
    exit 1
fi

# Change to the source directory.
cd "$source_dir" || { echo "Error: Unable to change directory to '$source_dir'."; exit 1; }

# Check for uncommitted changes; git diff --quiet returns non-zero if there are differences.
if ! git diff --quiet src/confopt; then
    echo "Error: Git working directory is not clean. Please commit or stash your changes before proceeding."
    exit 1
fi

if [ -n "$(git status --porcelain src/confopt)" ]; then
    echo "Error: There are uncommitted changes in src/confopt. Please commit or stash your changes before proceeding."
    exit 1
fi

# --- Check for untracked files in src/confopt ---

UNTRACKED=$(git ls-files --others --exclude-standard src/confopt)
if [ -n "$UNTRACKED" ]; then
    echo "Warning: The following files in 'src/confopt' are untracked by Git and will not be copied:"
    echo "$UNTRACKED"
fi

# --- Retrieve Git branch and commit information ---

BRANCH=$(git rev-parse --abbrev-ref HEAD)
COMMIT=$(git rev-parse HEAD)
MESSAGE=$(git log -1 --pretty=%B)
# --- Copy only the tracked files from src/confopt to the destination/confopt folder ---

# Create an archive of the tracked files under src/confopt and extract it into TARGET_DIR,
# stripping the leading 'src/confopt' components.
git archive HEAD src/confopt | tar -x --strip-components=2 -C "$TARGET_DIR"

# --- Write Git branch and commit info to a file outside the confopt folder ---

{
    echo "Branch: $BRANCH"
    echo "Commit: $COMMIT"
    echo "Message: $MESSAGE"
} > "$DEST_DIR/info"

echo "Tracked files from 'src/confopt' have been successfully copied to '$TARGET_DIR'."
echo "Git info stored in '$DEST_DIR/info"
