#!/bin/bash

echo "Creating virtual environment..."
python3.11 -m venv --clear venv

echo "Bootstrapping pip..."
venv/bin/python3.11 -m ensurepip --upgrade

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
venv/bin/python3.11 -m pip install -r requirements.txt

echo "Installing the package..."
venv/bin/python3.11 -m pip install -e .

echo "Installation complete."

