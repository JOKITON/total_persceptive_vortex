#!/bin/bash

# Create a virtual environment
if [ -d ".venv/activate.lock" ]; then
	echo "Virtual environment does not exist..."
else
	rm .venv/activate.lock
	deactivate
fi
