# import OpenAI API Key
from dotenv import load_dotenv
import os
load_dotenv()


# LLM and Embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load Document, split document & vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Abstraction for message roles
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Loading api key
openai_key = os.getenv("OPENAI_API_KEY")

# print(openai_key)

# First let's load the pdf
loader = PyPDFLoader('./storm.pdf')
pages_list = loader.load()

# print(pages_list)

# Splitting the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
pdf_chunks = splitter.split_documents(pages_list)

# Embedding Model
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', api_key= openai_key)

# Storing in Qdrant VectorDB
vector_db = Qdrant.from_documents(documents=pdf_chunks, embedding=embedding_model, distance="cosine", location=":memory:")


# LLM
llm = ChatOpenAI(model="gpt-4o", api_key=openai_key, temperature=0.4)


from fastapi import FastAPI

# # making the api async
import asyncio

app = FastAPI()


# This list is being used for chat history
messages = []

system_prompt = ''' You are an AI agent who answers user's question about a PDF. You will be given relevant chunks from the pdf that may help answer the user's question.\n
Instructions to Follow:\n
1. If the context doesn't contain the answer to the user question then politely tell him that you do not have the answer for it.\n
2. Before giving the answer critically analyze the given context to find any relevant information for the user's question.\n
3. Do not create any false information, if you do not have the answer then politely tell that.\n
4. The response should not be too long but specific to the question and should be brief.\n
'''

messages.append(SystemMessage(content=f"{system_prompt}"))




async def join_chunks(chunks):
    text = ''
    for i, chunk in enumerate(chunks):
        if i != 0: 
            text += '\n\n'
        text += chunk.page_content

    print("Text: \n",text)

    return text




async def get_similar_chunks(query:str, k:int):

    # Getting the current event loop
    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(
        None, lambda: vector_db.similarity_search(query=query, k=k)
    )



# Response and Request classes
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query:str = Field(..., example="What is this pdf about?")
    k:int = Field(default=1, ge=1, le=5)


class QueryResponse(BaseModel):
    answer:str





# The endpoint that takes user's query and get's context based response from the LLM
@app.post('/query')
async def answer_query(request:QueryRequest):

    try:

        retrieved_chunks = await get_similar_chunks(query=request.query, k=request.k)
        print("Retrieved Chunks: ",retrieved_chunks)


        context = await join_chunks(retrieved_chunks)


        messages.append(HumanMessage(content=f"Use the following context to answer user's question.\n\n Context: {context} \n\n User's Question: {request.query}"))


        response = await llm.ainvoke(messages)

        messages.append(AIMessage(content=response.content))

        return response.content

    except Exception as e:
        print(f"Getting error {e}")










