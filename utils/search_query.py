def generate_search_query(question, category):
    """
    Uses the LLM to extract the core medical concept to search for.
    Why? Because the raw question is too long for a wiki search.
    """
    prompt = [
        {"role": "system", "content": "You are a search engine optimizer for medical data."},
        {"role": "user", "content": f"""
        Extract the SINGLE most important medical concept or disease from this question to search on Wikipedia.
        
        Question: {question}
        Category: {category}
        
        Output ONLY the search term. Do not output a sentence.
        Example: 
        Input: "A 45-year-old man with crushing chest pain..."
        Output: Myocardial infarction
        """}
    ]
    # Small max_tokens because we just want a keyword
    output = pipe(prompt, max_new_tokens=20)
    search_term = output[0]['generated_text'][-1]['content'].strip()
    return search_term