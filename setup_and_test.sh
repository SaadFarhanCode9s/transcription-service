#!/usr/bin/env bash

# ==============================================================================
# Speech-to-Text Transcription Service — Installation & Verification Script
# ==============================================================================
#
# This script:
# 1. Verifies/installs system dependencies (FFmpeg, libmagic).
# 2. Creates a Python virtual environment (handling debian/ubuntu specific packages).
# 3. Installs all python dependencies from requirements.txt.
# 4. Executes the test suite via pytest to verify the installation.
#
# Safe to run multiple times (idempotent).
# ==============================================================================

set -euo pipefail

# Text formatting helper functions
info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

warn() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

# ------------------------------------------------------------------------------
# 1. System Dependency Checks & Installation
# ------------------------------------------------------------------------------
info "Checking system dependencies..."

# Determine package manager
HAS_APT=false
if command -v apt-get &>/dev/null; then
    HAS_APT=true
fi

# Detect missing binaries/libraries
NEEDS_FFMPEG=false
NEEDS_LIBMAGIC=false
NEEDS_VENV_PKG=false

if ! command -v ffmpeg &>/dev/null; then
    NEEDS_FFMPEG=true
fi

# libmagic1 check (look in typical library paths or check python-magic loadability)
if ! python3 -c "import magic" &>/dev/null && [ "$HAS_APT" = true ]; then
    # We will install libmagic1 if we install system packages
    NEEDS_LIBMAGIC=true
fi

# Check if python3-venv / ensurepip is present
if ! python3 -m venv --help &>/dev/null; then
    NEEDS_VENV_PKG=true
fi

# Install system dependencies if required and permissions exist
if [ "$NEEDS_FFMPEG" = true ] || [ "$NEEDS_LIBMAGIC" = true ] || [ "$NEEDS_VENV_PKG" = true ]; then
    warn "Missing system dependencies detected."
    
    if [ "$HAS_APT" = true ]; then
        info "Installing system packages via apt-get..."
        
        # Build apt command
        APT_PKGS=""
        if [ "$NEEDS_FFMPEG" = true ]; then APT_PKGS="$APT_PKGS ffmpeg"; fi
        if [ "$NEEDS_LIBMAGIC" = true ]; then APT_PKGS="$APT_PKGS libmagic1"; fi
        if [ "$NEEDS_VENV_PKG" = true ]; then
            # Match python3 version dynamically
            PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            APT_PKGS="$APT_PKGS python3-pip python3-venv python3.${PY_VER}-venv"
        fi
        
        # Run apt with sudo if not root
        if [ "$EUID" -ne 0 ]; then
            if command -v sudo &>/dev/null; then
                info "Requesting sudo permissions to run: apt-get install -y $APT_PKGS"
                sudo apt-get update && sudo apt-get install -y $APT_PKGS
            else
                error "Sudo is not available. Please manually run as root: apt-get update && apt-get install -y $APT_PKGS"
                exit 1
            fi
        else
            apt-get update && apt-get install -y $APT_PKGS
        fi
    else
        error "Non-debian/ubuntu platform detected and dependencies are missing."
        error "Please install ffmpeg, libmagic1, and python3-venv manually before proceeding."
        exit 1
    fi
else
    success "All system dependencies (FFmpeg, libmagic, venv module) are available."
fi

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup
# ------------------------------------------------------------------------------
info "Setting up Python virtual environment..."

# Clean up previous broken venv if exists
if [ -d ".venv" ]; then
    info "Removing existing .venv directory..."
    rm -rf .venv
fi

# Attempt to create virtual environment
if python3 -m venv .venv; then
    info "Virtual environment created successfully."
    # Activate virtual environment
    # shellcheck disable=SC1091
    source .venv/bin/activate
    info "Upgrading pip..."
    pip install --upgrade pip
else
    warn "Failed to create virtual environment via python3 -m venv."
    warn "Falling back to user-level installation (--user) using system python..."
    
    # Define a helper function to run commands with --user and --break-system-packages
    # to avoid PEP 668 restrictions on modern systems.
    run_pip_fallback() {
        python3 -m pip install --user --break-system-packages "$@"
    }
fi

# ------------------------------------------------------------------------------
# 3. Installing Python Dependencies
# ------------------------------------------------------------------------------
info "Installing Python dependencies..."

if [ -n "${VIRTUAL_ENV:-}" ]; then
    # Venv is active, standard pip install
    pip install -r requirements.txt
    # Also ensure python-magic is installed in the venv
    pip install python-magic
else
    # Fallback to system-level package installation
    run_pip_fallback -r requirements.txt
    run_pip_fallback python-magic
fi

success "Dependencies successfully installed."

# ------------------------------------------------------------------------------
# 4. Running the Pytest Suite
# ------------------------------------------------------------------------------
info "Running test suite to verify installation..."

# Formulate pytest command
PYTEST_CMD="python3 -m pytest tests/ -v --tb=short --no-header -p no:cacheprovider"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    # Run in virtual environment context
    eval "$PYTEST_CMD"
else
    # Run in system Python user-site context, ensuring local user binary path is in PATH
    export PATH="$HOME/.local/bin:$PATH"
    eval "$PYTEST_CMD"
fi

success "All checks and tests passed successfully! The service is ready."
info "To run the service locally, run:"
info "  python3 main.py"
