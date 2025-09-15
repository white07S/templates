#!/bin/bash
TOKEN=$(az account get-access-token \
  --resource https://cognitiveservices.azure.com/ \
  --query accessToken -o tsv)

export OPENAI_API_TYPE=azure
export OPENAI_API_BASE="https://<your-resource-name>.openai.azure.com/"
export OPENAI_API_VERSION="2024-08-01-preview"
export OPENAI_API_KEY="Bearer $TOKEN"

codex "$@"
