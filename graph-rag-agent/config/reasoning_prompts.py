from config.settings import KB_NAME

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
MAX_SEARCH_LIMIT = 5

REASON_PROMPT = (
        f"You are a reasoning assistant that can use search tools to search for {KB_NAME} related questions, helping you accurately answer user questions. You have special tools:\n\n"
        f"- To perform a search: please write {BEGIN_SEARCH_QUERY} your query content {END_SEARCH_QUERY}.\n"
        f"Then, the system will search and analyze relevant content, then provide useful information in the format {BEGIN_SEARCH_RESULT} ...search results... {END_SEARCH_RESULT}.\n\n"
        f"You can repeat the search process as needed. The maximum search limit is {MAX_SEARCH_LIMIT} times.\n\n"
        "After getting all the information you need, continue your reasoning.\n\n"
        "-- Example 1 --\n"
        "Question: \"Are the directors of the movies 'Jaws' and 'Casino Royale' from the same country?\"\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Who is the director of 'Jaws'?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nThe director of 'Jaws' is Steven Spielberg...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information.\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Where is Steven Spielberg from?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nSteven Allan Spielberg is an American filmmaker...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information...\n\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Who is the director of 'Casino Royale'?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\n'Casino Royale' is a 2006 spy film directed by Martin Campbell...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information...\n\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}Where is Martin Campbell from?{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nMartin Campbell is a New Zealand film director...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information...\n\n"
        "Assistant:\n"
        "Now I have enough information to answer this question.\n\n"
        "**Answer**: No, the director of 'Jaws', Steven Spielberg, is from the United States, while the director of 'Casino Royale', Martin Campbell, is from New Zealand. They are not from the same country.\n\n"
        "-- Example 2 --\n"
        "Question: \"What are the scholarship policies for excellent students?\"\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}scholarship policies for excellent students{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nScholarship policies for excellent students include merit-based awards and need-based assistance...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information.\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}specific standards for excellent student scholarships{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nExcellent student scholarship standards require top 10% academic performance...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information...\n\n"
        "Assistant:\n"
        f"    {BEGIN_SEARCH_QUERY}scholarship application procedures for students{END_SEARCH_QUERY}\n\n"
        "User:\n"
        f"    {BEGIN_SEARCH_RESULT}\nScholarship application procedures include submitting academic transcripts and recommendation letters...\n{END_SEARCH_RESULT}\n\n"
        "Continue reasoning with new information...\n\n"
        "Assistant:\n"
        "Now I have enough information to answer this question.\n\n"
        "**Answer**: Scholarship policies for excellent students mainly include two aspects: 1) Merit-based awards: requiring top 10% academic performance; 2) Application procedures: submitting academic transcripts and recommendation letters.\n\n"
        "**Remember**:\n"
        f"- You have a knowledge base to search, just provide appropriate search queries.\n"
        f"- Use {BEGIN_SEARCH_QUERY} to request database search, ending with {END_SEARCH_QUERY}.\n"
        "- Query language must use the same language as 'question' or 'search results'.\n"
        "- If you can't find useful information, rewrite the search query to use fewer and more precise keywords.\n"
        "- After completing the search, continue your reasoning to answer the question.\n\n"
        'Please answer the following question. You should think step by step to solve it.\n\n'
    )

RELEVANT_EXTRACTION_PROMPT = """**Task Description:**

    You need to read and analyze the searched content based on the following inputs: **previous reasoning steps**, **current search query**, and **searched content**. Your goal is to extract useful information from the **searched content** that is relevant to the **current search query**, and seamlessly integrate this information into the **previous reasoning steps** to continue reasoning for the original question.

    **Guidelines:**

    1. **Analyze the searched content:**
    - Carefully review the content of the search results.
    - Identify factual information related to the **current search query** that can help the reasoning process answer the original question.

    2. **Extract relevant information:**
    - Select information that can directly advance the **previous reasoning steps**.
    - Ensure the extracted information is accurate and relevant.

    3. **Output format:**
    - **If the search content provides useful information for the current query:** Present the information starting with `**Final Information**` as shown below.
    - The output language **must** be consistent with the language of the 'search query' or 'search content'.\n"
    **Final Information**

    [Useful information]

    - **If the search content provides no useful information for the current query:** Output the following text:

    **Final Information**

    No helpful information found.

    **Input:**
    - **Previous reasoning steps:**  
    {prev_reasoning}

    - **Current search query:**  
    {search_query}

    - **Searched content:**  
    {document}

    """

SUB_QUERY_PROMPT = """To answer this question more comprehensively, please break down the original question into at most three sub-questions. Return a list of strings.
        If this is a very simple question that doesn't need to be broken down, keep only one original question in the list.

        Please ensure that each sub-question is clear, can be answered independently, and is directly related to the original question.

        Original question: {original_query}

        Example input:
        "What are the scholarship policies for excellent students?"

        Example output:
        [
            "What are the academic requirements for excellent student scholarships?",
            "What are the application procedures for excellent student scholarships?",
            "What are the scholarship amounts and benefits for excellent students?"
        ]

        Please provide your answer in Python list format:
        """

FOLLOWUP_QUERY_PROMPT = """Based on the original query and retrieved information, determine if additional search queries are needed. If further research is needed, provide at most 2 search queries. If no further research is needed, return an empty list.

        Original query: {original_query}

        Retrieved information: 
        {retrieved_info}

        Please consider the following factors:
        1. Whether the existing information completely answers the original query
        2. Whether there are information gaps or unresolved questions
        3. Whether more specific or recent information is needed

        Please return only valid Python string list format, without any other text.
        """

FINAL_ANSWER_PROMPT = """Based on the following thinking process and retrieved information, answer the user's question. Provide a detailed, accurate, and comprehensive answer with relevant sources cited.

        User question: "{query}"

        Retrieved information:
        {retrieved_content}

        Thinking process:
        {thinking_process}

        Please generate a comprehensive final answer. You don't need to explain your thinking process, just give the conclusion directly. Ensure the answer is clear, well-structured, and includes relevant important information.
        
        Format requirements:
        1. Use concise and clear language
        2. Use appropriate title structure to organize information
        3. Don't say "according to retrieved information" and similar expressions
        4. Give definitive answers directly, don't use uncertain expressions like "might", "perhaps" (unless there is genuine uncertainty)
        """