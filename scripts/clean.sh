#!/bin/bash

# Create a virtual environment
if [ -d ".venv" ]; then
	rm -rf .venv
else
	pecho "Virtual environment does not exist..."
fi