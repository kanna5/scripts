#!/bin/bash
set -e -o pipefail

source ~/.config/openai/env

curl -sSLf https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_APIKEY" | jq -r .data[].id | sort -u
