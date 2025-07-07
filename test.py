from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List


from dotenv import load_dotenv
import os
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0.4)


class QueryResponse(BaseModel):
    summary: str
    key_entities: List
    sentiment: str


query = '''In a landmark move, the city of Greenhaven has committed to transitioning to 100% renewable energy by 2035.
The mayor, Lisa Tran, announced the plan in front of a large crowd gathered at City Hall. The initiative includes major investments in solar and wind infrastructure, as well as incentives for residents to install rooftop solar panels.
Environmental advocacy groups such as Green Future Coalition applauded the decision, calling it a bold and necessary step in the fight against climate change.
However, some local businesses expressed concern over the potential rise in energy costs and the feasibility of implementing the new grid technologies in a short time frame.
Despite mixed reactions, the city council passed the resolution unanimously, marking a major milestone in Greenhaven’s environmental policy.
'''

parser = JsonOutputParser(pydantic_object=QueryResponse)

system_prompt = '''
You are an AI agent trained to analyze and summarize text.\n
Your task is to process a given passage of text and provide a structured response in JSON format.\n

Please follow these instructions:\n\n
1. Summarize the Text:\n   
- Provide a concise summary of the passage in bullet points. Each bullet point should capture a key aspect or idea from the text.\n\n
2. Identify Key Entities:\n   
- Identify three key entities mentioned in the passage. For each entity, provide a brief description of their role or significance within the context of the text.\n\n
3. Sentiment Analysis:\n   
- Analyze the overall sentiment of the passage and categorize it as \"positive,\" \"negative,\" or \"neutral.\" Provide a brief explanation supporting your sentiment classification.\n\n

'''

prompt = PromptTemplate(
    template="{system_prompt}\n{format_instructions}\n{query}\n",
    input_variables=["query", system_prompt],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

response = chain.invoke({"query": query, 'system_prompt': system_prompt})

print(response)