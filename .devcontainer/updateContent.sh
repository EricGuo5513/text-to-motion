#!/bin/bash

# Create symlink to checkpoints so our scripts can use them relative to the workspace
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ln -sf /checkpoints ${SCRIPT_DIR}/checkpoints