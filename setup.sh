#!/bin/bash

# Define the target virtual environment folder name
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
PYTHON_EXE="python3"
REQUIRED_VERSION="3.10.4"

# 1. Check if the requested Python command is actually installed on the system
if ! command -v "$PYTHON_EXE" &> /dev/null; then
    echo "Error: $PYTHON_EXE is not installed on your system."
    return 1 2>/dev/null || exit 1
fi

# 2. Strict Version Check
# We run a tiny Python script inline to print just the version number (e.g., "3.10.4")
ACTUAL_VERSION=$("$PYTHON_EXE" -c 'import platform; print(platform.python_version())')

if [ "$ACTUAL_VERSION" != "$REQUIRED_VERSION" ]; then
    echo "Error: Strict version requirement failed."
    echo "Required: $REQUIRED_VERSION"
    echo "Found:    $ACTUAL_VERSION"
    echo "Please install Python $REQUIRED_VERSION and ensure it is in your PATH."
    return 1 2>/dev/null || exit 1
fi

echo "Python version $ACTUAL_VERSION verified. Proceeding..."

# 3. Check if the virtual environment folder already exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON_EXE" -m venv "$VENV_DIR"
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 4. Detect the OS to locate the correct activation script path
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Path for Git Bash / Windows
    echo "Activating virtual environment (Windows/Git Bash)..."
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    # Path for macOS / Linux
    echo "Activating virtual environment (macOS/Linux)..."
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Activation script not found."
    return 1 2>/dev/null || exit 1
fi

# 5. Install requirements if they exist
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
else
    echo "Warning: $REQ_FILE not found. Skipping library installation."
fi

# 6. Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Setup complete!"