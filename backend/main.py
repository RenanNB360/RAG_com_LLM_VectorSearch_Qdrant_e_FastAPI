import os
import rag
from fastapi  import FastAPI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import uvicorn

_ = load_dotenv(find_dotenv())
nvidia_api = os.environ.get('api_nvidia_llama3_70b')

class Item(BaseModel):
    query: str

model_name = 'sentence-transformers/msmarco-bert-base-dot-v5'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(model_name= model_name, model_kwargs= model_kwargs, encode_kwargs= encode_kwargs)

use_nvidia_api = False

if nvidia_api:

    client_ai = OpenAI(base_url= 'https://integrate.api.nvidia.com/v1', api_key= nvidia_api)
    use_nvidia_api = True

else:
    print('Não foi possível localizar a chave api.')

client = QdrantClient('http://localhost:6333')
collection_name = 'VectorDB'
qdrant = QdrantVectorStore(client= client, collection_name= collection_name, embedding= hf)


app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'LLM com RAG e Qdrant com Banco Vetorial'}

@app.post('/api')
async def api(item: Item):
    query = item.query
    search_result = qdrant.similarity_search(query= query, k= 10)

    list_res = []
    context = ''
    mappings = {}

    for i, res in enumerate(search_result):
        context += f'{i}\n{res.page_content}\n\n'
        mappings[i] = res.metadata.get('path')
        list_res.append({'id': i, 'path': res.metadata.get('path'), 'content': res.page_content})
                        
    rolemsg = {'role': 'system',
               'content': (
        'Responda à pergunta do usuário usando documentos fornecidos no contexto. '
        'No contexto estão documentos que devem conter uma resposta. Sempre faça referência '
        'ao ID do documento (entre colchetes, por exemplo [0],[1]) do documento que foi usado '
        'para fazer uma consulta. Use quantas citações e documentos forem necessários para responder à pergunta.'
    )}

    messages = [rolemsg, {'role': 'user', 'content': f'Documents:\n{context}\n\nQuestion: {query}'}]

    if use_nvidia_api:
        resposta = client_ai.chat.completions.create(model= 'meta/llama3-70b-instruct',
                                                     messages= messages,
                                                     temperature= 0.5,
                                                     top_p = 1,
                                                     max_tokens = 1024,
                                                     stream = False)
        
        response = resposta.choices[0].message.content

    else:
        print('Não é possível usar um LLM.')
    
    return {'context': list_res, 'answer': response}

"""if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)"""
