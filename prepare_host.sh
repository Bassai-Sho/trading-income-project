#!/usr/bin/env bash
# Redirect to unified setup.sh
exec "$(dirname "$0")/setup.sh" "$@"
