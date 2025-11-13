#!/bin/env bash
# Make sure the working directory is the root of the project
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit