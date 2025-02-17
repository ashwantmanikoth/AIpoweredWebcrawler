from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, ConfigDict, model_validator
import warnings
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Search API using SerpAPI."""

    api_key: str = Field(..., env="SERP_API_KEY")
    max_results: int = 2
    safesearch: str = "off"
    time: Optional[str] = "y"
    source: str = "text"
    
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that python package exists in environment."""
        try:
            from serpapi import GoogleSearch  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import serpapi python package. "
                "Please install it with `pip install google-search-results`."
            )
        return values

    def _google_search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """Run query through Google Search API and return results."""
        from serpapi import GoogleSearch

        params = {
            "q": query,
            "api_key": self.api_key,
            "num": max_results or self.max_results,
            "safe": self.safesearch,
            "gl": "us"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        results = results.get("organic_results", [])
        
        return [{
            "snippet": r.get("snippet", ""),
            "title": r.get("title", ""),
            "link": r.get("link", "")
        } for r in results] if results else []

    def run(self, query: str) -> str:
        """Run query through Google Search and return concatenated results."""
        results = self._google_search(query)
        if not results:
            return "No good Google Search Result was found"
        return " ".join(r["snippet"] for r in results)

    def results(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Run query through Google and return metadata."""
        return self._google_search(query, max_results=max_results)

class GoogleSearchInput(BaseModel):
    """Input for the Google search tool."""
    query: str = Field(description="Search query to look up")

class GoogleSearchRun(BaseTool):
    """Google Search tool using SerpAPI."""
    name: str = "google_search"
    description: str = (
        "A wrapper around Google Search using SerpAPI. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: GoogleSearchAPIWrapper = Field(default_factory=GoogleSearchAPIWrapper)
    args_schema: Type[BaseModel] = GoogleSearchInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

class GoogleSearchResults(BaseTool):
    """Tool that queries the Google Search API and gets back json string."""
    name: str = "google_results_json"
    description: str = (
        "A wrapper around Google Search using SerpAPI. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON string array of the query results."
    )
    max_results: int = Field(alias="num_results", default=4)
    api_wrapper: GoogleSearchAPIWrapper = Field(default_factory=GoogleSearchAPIWrapper)
    args_schema: Type[BaseModel] = GoogleSearchInput
    keys_to_include: Optional[List[str]] = None
    results_separator: str = ", "

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        res = self.api_wrapper.results(query, self.max_results)
        res_strs = [
            ", ".join(
                [
                    f"{k}: {v}" for k, v in d.items() if not self.keys_to_include or k in self.keys_to_include
                ]
            ) for d in res
        ]
        return self.results_separator.join(res_strs)


def GoogleSearchTool(*args: Any, **kwargs: Any) -> GoogleSearchRun:
    """
    Deprecated. Use GoogleSearchRun instead.

    Args:
        *args:
        **kwargs:

    Returns:
        GoogleSearchRun
    """
    warnings.warn(
        "GoogleSearchTool will be deprecated in the future. "
        "Please use GoogleSearchRun instead.",
        DeprecationWarning,
    )
    return GoogleSearchRun(*args, **kwargs)