#!/bin/bash
TOKEN=$(az account get-access-token \
  --resource https://cognitiveservices.azure.com/ \
  --query accessToken -o tsv)

export OPENAI_API_TYPE=azure
export OPENAI_API_BASE="https://<your-resource-name>.openai.azure.com/"
export OPENAI_API_VERSION="2024-08-01-preview"
export OPENAI_API_KEY="Bearer $TOKEN"

codex "$@"


# pip install openai azure-identity
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Option A: AAD token (no API key)
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    azure_endpoint="https://<your-resource>.openai.azure.com/",
    api_version="2024-08-01-preview",
    azure_ad_token_provider=token_provider,
)

# If you have an API key instead, use: client = AzureOpenAI(
#   azure_endpoint="https://<your-resource>.openai.azure.com/",
#   api_key="<AZURE_OPENAI_API_KEY>",
#   api_version="2024-08-01-preview"
# )

models = client.models.list()
for m in models.data:
    # m.id == deployment name you pass as `model`
    print({
        "deployment_name": m.id,
        "created": m.created,
        "owned_by": m.owned_by,
        "object": m.object,
    })
