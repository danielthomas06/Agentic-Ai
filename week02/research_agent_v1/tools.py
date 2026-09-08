import requests

from pydantic import BaseModel


class SearchArguments(BaseModel):
    topic: str


def search_wikipedia(topic: str) -> dict:
    url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
        + requests.utils.quote(topic)
    )

    headers = {
        "User-Agent": "Agentic-AI-Research-Agent/1.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 404:
            return {
                "success": False,
                "error": f"No Wikipedia page found for '{topic}'.",
            }

        if response.status_code == 403:
            return {
                "success": False,
                "error": "Wikipedia rejected the request (HTTP 403).",
            }

        response.raise_for_status()

        data = response.json()

        return {
            "success": True,
            "title": data.get("title"),
            "summary": data.get("extract"),
            "url": (
                data.get("content_urls", {})
                .get("desktop", {})
                .get("page")
            ),
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Wikipedia request failed: {e}",
        }

search_wikipedia_tool = Tool(
    name="search_wikipedia",
    description="Search Wikipedia for a topic.",
    function=search_wikipedia,
    argument_model=SearchArguments,
)