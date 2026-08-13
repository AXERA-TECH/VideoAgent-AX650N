

GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."





PROMPTS[
    "videorag_response"
] = """---Role---
You are an intelligent video content assistant. Your task is to answer user queries based **strictly** on the provided Video Data and Text Chunks.

---Data Sources---
1. **Video Data**: Contains specific video segments, names, and timestamps.
   {video_data}
2. **Text Chunks**: General context or transcript information.
   {chunk_data}

---Instructions---
1. **Synthesize**: Combine information from both sources to answer the user's question comprehensively.
2. **Strict Grounding**: 
   - Do NOT make up facts. 
   - If the provided data is insufficient, clearly state: "I don't have enough information to answer that based on the provided data."
3. **Citation (Crucial)**: 
   - When using info from Video Data, you MUST cite it using the format `[x]` in the text.
   - List full details in a "#### References" section at the end.

---Citation Format---
**In-text:** ...highlighted the effects of deforestation [1].
**Reference List:**
#### References
[1] **Video Name**, Start: 05:30, End: 08:00

---Output Constraints---
- **NO Thinking Process**: Do NOT output `` or `</think>` tags. Perform your reasoning silently and output ONLY the final response.
- **Format**: Markdown.
- **Tone**: Professional and objective.

---Example Output---
The protagonist appears in the red suit during the chase [1].

#### References
# [1] video_name_1, 05:30, 08:00  
# [2] video_name_2, 25:00, 28:00 

---Goal---
Generate a response that answers the user's question based strictly on the provided data, ensuring all video sources are cited with their specific timestamps.
"""






PROMPTS[
    "keywords_extraction"
] = """- Role -
You are a strict keyword extraction tool. Your output must contain ONLY the extracted keywords.

- Constraints -
1.  **Strictly Prohibited**: Do NOT output tags like `

`, `<answer>`, or any reasoning steps.
2.  **No Explanations**: Do not output "Here are the keywords" or any conversational text.
3.  **Format**: Output a single line of comma-separated keywords.
4.  **Clean Output**: If you think you need to reason, do it silently. Only print the final keywords.

- Goal -
Extract relevant keywords from the query. Focus on entities, actions, and specific attributes. Remove stop words (e.g., "the", "is", "when", "video") unless necessary.

######################
- Examples -
######################

Question: Which animal does the protagonist encounter in the forest scene?
################
Output:
animal, protagonist, forest scene, encounter

Question: What is the weather like?
################
Output:
weather

#############################
- Real Data -
######################
Question: {input_text}
######################
Output:
"""