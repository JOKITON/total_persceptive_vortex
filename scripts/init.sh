#!/bin/bash

export os_choice=1

# Create a virtual environment
if [ -d ".venv" ]; then
	echo "Virtual environment already exists..."
else
	# Set the appropriate environment variables based on the user's choice
	if [ $os_choice -eq 1 ]; then
		echo "You chose Mac."
		python3.12 -m venv .venv
	elif [ $os_choice -eq 2 ]; then
		echo "You chose Windows/Linux."
		python3.12 -m venv .venv
    elif [ $os_choice -eq 3 ]; then
		echo "You chose 42 Linux."
		virtualenv -p /sgoinfre/students/jaizpuru/homebrew/bin/python3.9 .venv
	else
		echo "Invalid choice. Exiting."
	fi
fi

if [ -f ".venv/activate.lock" ]; then
	echo "Virtual enviroment already running..."
else
	# Create a lock file
	touch .venv/activate.lock

	# Set the appropriate environment variables based on the user's choice
	if [ $os_choice -eq 1 ]; then
		echo "You chose Mac."
		# Activate the virtual environment
		source .venv/bin/activate
		pip install --upgrade pip
		pip install -r requirements.txt
	elif [ $os_choice -eq 2 ]; then
		echo "You chose Windows/Linux."
		# Activate the virtual environment
		source .venv/bin/activate
		pip install --upgrade pip
		pip install -r requirements.txt
    elif [ $os_choice -eq 3 ]; then
		echo "You chose 42 Linux."
		.venv/bin/pip install --upgrade pip
		.venv/bin/pip install -r requirements.txt
	else
		echo "Invalid choice. Exiting."
	fi
fi
