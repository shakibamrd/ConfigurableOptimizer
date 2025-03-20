#!/bin/bash
set -e

# --- Configuration ---

# Determine the directory where this script is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${SCRIPT_DIR}/config.cfg"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# Source the config file (it should define a variable named 'source_dir_local')
source "$CONFIG_FILE"

if [ -z "$source_dir_local" ]; then
    echo "Error: 'source_dir_local' is not defined in $CONFIG_FILE"
    exit 1
fi

# Ensure that the src/confopt folder exists within the source directory.
SRC_FOLDER="${source_dir_local}/src/confopt"
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

# --- Ensure source_dir_local is a Git repository and has no local changes ---

if [ ! -d "${source_dir_local}/.git" ]; then
    echo "Error: '$source_dir_local' is not a Git repository ('.git' folder not found)."
    exit 1
fi

# Change to the source directory.
cd "$source_dir_local" || { echo "Error: Unable to change directory to '$source_dir_local'."; exit 1; }

# Check for uncommitted changes; git diff --quiet returns non-zero if there are differences.
if ! git diff --quiet src/confopt; then
    echo "Error: Git working directory is not clean. Please commit or stash your changes before proceeding."
    exit 1
fi

if ! git diff --quiet copycon/*.py copycon/*.sh; then
    echo "Error: Git working directory copycon/ is not clean. Please commit or stash your changes before proceeding."
    exit 1
fi

if [ -n "$(git status --porcelain src/confopt)" ]; then
    echo "Error: There are uncommitted changes in src/confopt. Please commit or stash your changes before proceeding."
    exit 1
fi


if [ -n "$(git status --porcelain copycon/*.sh copycon/*.py)" ]; then
    echo "Error: There are uncommitted changes in copycon/. Please commit or stash your changes before proceeding."
    exit 1
fi

# Define the target directory as a subdirectory 'confopt' in the destination directory.
TARGET_DIR="${DEST_DIR}/confopt"

# If the target directory already exists, ask for confirmation to overwrite it.
if [ -d "$TARGET_DIR" ]; then
    read -p "Directory '$DEST_DIR' already exists. Do you want to overwrite it? (y/n): " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        rm -rfv "$TARGET_DIR"
        rm -rfv "$DEST_DIR/info"
        rm -rfv "$DEST_DIR/launch.py"
        rm -rfv "$DEST_DIR/run_exp.py"
		echo ""
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

mkdir -p "$DEST_DIR"

# Create the target directory.
mkdir -p "$TARGET_DIR"

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
DATE=$(git show -s --format=%cd --date=format:'%A, %B %d, %Y %H:%M:%S')
# --- Copy only the tracked files from src/confopt to the destination/confopt folder ---

# Create an archive of the tracked files under src/confopt and extract it into TARGET_DIR,
# stripping the leading 'src/confopt' components.
git archive HEAD src/confopt | tar -x --strip-components=2 -C "$TARGET_DIR"

# --- Write Git branch and commit info to a file outside the confopt folder ---

{
    echo "Branch: $BRANCH"
    echo "Commit: $COMMIT"
    echo "Message: $MESSAGE"
    echo "Date: $DATE"
} > "$DEST_DIR/info"

echo "Tracked files from 'src/confopt' have been successfully copied to '$TARGET_DIR'."
echo "Git info stored in '$DEST_DIR/info"

cp -v ${source_dir_local}/copycon/launch_supernet_search.py ${DEST_DIR}
cp -v ${source_dir_local}/copycon/launch_model_train.py ${DEST_DIR}
cp -v ${source_dir_local}/copycon/run_supernet_search.py ${DEST_DIR}
cp -v ${source_dir_local}/copycon/run_all.sh ${DEST_DIR}
cp -v ${source_dir_local}/scripts/benchsuite_experiments/run_model_train.py ${DEST_DIR}
cp -v ${CONFIG_FILE} ${DEST_DIR}

# copy benchsuite.py to the destination directory, if it exists
if [ -f $source_dir_local/benchsuite.py ]; then
    echo "Found benchsuite.py in source directory. Copying it to '$DEST_DIR'."
    cp $source_dir_local/benchsuite.py ${DEST_DIR}/benchsuite.py
fi
