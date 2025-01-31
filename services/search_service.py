from utils.DuckDuckGo import DuckDuckGoSearchResults
from langchain.schema import Document
import json, re

def extract_json_from_text(text):
    """Extract JSON part from a mixed text."""
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group()
    raise ValueError("No JSON content found in the provided text")

def generate_prompt(query):
    return f"""
        You are an intelligent assistant that helps users create search queries based on their input. When a user provides a query, your task is to identify relevant search queries that can be used for web searches. If the user’s query is too long or complex, break it down into multiple simpler search queries. 
        
        Your output should always be in the following JSON format:
        {{
        "search_query": ["search1", "search2", ...]
        }}
        Make sure to maintain the specified format strictly and provide clear and concise search queries.
        User query: "{query}"
        """

def SearchLoader(query, num_results, llm):
    messages = [
        (
            "system", "You are a helpful assistant that returns output in json format"
         ),
          (
              "human", generate_prompt(query)
              )]
    try:
        results = llm.invoke(messages).content
        print("results: ", results)
    except Exception as e:
        raise RuntimeError(f"Error invoking LLM: {e}")


    try:
        # Extract and parse JSON from the raw LLM output
        json_text = extract_json_from_text(results)
        js = json.loads(json_text)
    except Exception as e:
        raise ValueError(f"Error extracting or decoding JSON from LLM output: {results}") from e
    src_query = js["search_query"]


    web_search = DuckDuckGoSearchResults(num_results=num_results)
    print("search result: ", web_search)

    documents = []
    for qry in src_query:
        res = web_search.invoke(qry)
        document = Document(page_content=res)
        documents.append(document)

    return documents, src_query
