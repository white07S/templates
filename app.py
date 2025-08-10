# pip install langchain langchain-openai pydantic

import os
from typing import Type
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_openai import AzureChatOpenAI  # or AzureOpenAI for completion-style
from langchain_core.language_models.chat_models import BaseChatModel


# ---------- 1) Example Pydantic schema ----------
class Incident(BaseModel):
    title: str = Field(..., description="Short, human-readable title")
    severity: Literal["low", "medium", "high"]
    tags: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="2–3 sentence summary")


# ---------- 2) Generic structured-output helper ----------
def run_structured(
    client: BaseChatModel,
    prompt: str,
    schema: Type[BaseModel],
) -> BaseModel:
    """
    Call an Azure OpenAI (LangChain) client and parse the response
    into the provided Pydantic schema using .with_structured_output().
    """
    structured = client.with_structured_output(schema)
    return structured.invoke(prompt)  # returns an instance of `schema`


# ---------- 3) Example usage ----------
if __name__ == "__main__":
    # Create the Azure client (chat models recommended for structured output)
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",                # your Azure deployment name
        api_version="2024-06-01",                      # your Azure API version
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0,
    )

    text = """
    We had an outage on the payments API yesterday. Card auths failed intermittently
    for 27 minutes. Root cause was an expired OAuth token that our rotation job
    missed due to a misconfigured cron expression. Impacted merchants: 14.
    """

    prompt = f"Extract an incident report from this text:\n\n{text}"

    result: Incident = run_structured(llm, prompt, Incident)
    print(result)              # Pretty __repr__
    print(result.model_dump()) # Dict form (Pydantic v2); use .dict() on v1
