import json, re
from utils.config import Config
from duckduckgo_search.exceptions import RatelimitException
from utils.GoogleSearchAPI import GoogleSearchAPIWrapper
from langchain.schema import Document

def extract_json_from_text(text):
    """Extract JSON part from a mixed text."""
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group()
    raise ValueError("No JSON content found in the provided text")

def search_query_generate_prompt(query):
    return f"""
        You are an intelligent assistant that helps users create search queries based on their input. When a user provides a query, your task is to identify relevant search queries that can be used for web searches. If the user’s query is too long or complex, break it down into multiple simpler search queries. 
        
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
              "human", search_query_generate_prompt(query)
              )]
    try:
        results = llm.invoke(messages).content
    except Exception as e:
        raise RuntimeError(f"Error invoking LLM: {e}")

    try:
        # Extract and parse JSON from the raw LLM output
        json_text = extract_json_from_text(results)
        js = json.loads(json_text)
        src_query = js["search_query"]
    except Exception as e:
        raise ValueError(f"Error extracting or decoding JSON from LLM output: {results}") from e
    
    #Duckduckgo search API 
    # web_search = DuckDuckGoSearchResults(num_results=num_results)

    #google search uing SERP API
    web_search = GoogleSearchAPIWrapper(api_key=Config.SERP_API_KEY, max_results=num_results)

    documents = []
    for qry in src_query:
        try:
            res = web_search.results(qry,max_results=num_results)

            if isinstance(res, list):  # Ensure it's a list of search results
                res_text = "\n".join(
                    [f"{r.get('title', '')}: {r.get('snippet', '')}" for r in res]
                )
            else:
                res_text = res  # If res is already a string

            document = Document(page_content=res_text)

            documents.append(document)
        except RatelimitException:
            print(f"Rate limit hit for query: {qry}. Skipping...")
            continue  # Skip this query and proceed to the next one
        except Exception as e:
            print(f"Error searching for query '{qry}': {e}")
            continue  # Skip this query and proceed to the next one    

    return documents, src_query
