#!/bin/bash
set -e

# Name of the script file to install (adjust if needed)
SCRIPT_NAME="copycon.sh"
INSTALL_NAME="copycon"

# Default system-wide installation directory.
INSTALL_DIR="/usr/local/bin"

# Check if running as root; if not, use a user-specific directory.
if [ "$(id -u)" -ne 0 ]; then
    echo "Not running as root. Installing to \$HOME/.local/bin instead."
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
fi

# Determine the directory where this installer is located.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Full path to the script to install.
SOURCE_SCRIPT="${SCRIPT_DIR}/${SCRIPT_NAME}"

# Check if the source script exists.
if [ ! -f "$SOURCE_SCRIPT" ]; then
    echo "Error: Source script '$SOURCE_SCRIPT' not found."
    exit 1
fi

# Copy the script to the target directory and make it executable.
echo "Installing '$INSTALL_NAME' to '$INSTALL_DIR'..."
cp "$SOURCE_SCRIPT" "$INSTALL_DIR/$INSTALL_NAME"
cp "$SCRIPT_DIR/copycon.cfg" "$INSTALL_DIR/copycon.cfg"
chmod +x "$INSTALL_DIR/$INSTALL_NAME"

echo "Installation complete. Please ensure '$INSTALL_DIR' is in your PATH."
echo "You can now run the command using:"
echo "  $INSTALL_NAME"

