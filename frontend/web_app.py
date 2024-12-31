import re
import os
import requests
import streamlit as st
import json


st.set_page_config(page_title= 'Projeto de RAG em Documentos', page_icon=':100:', layout= 'centered')

st.title('_:green[Busca com IA Generativa e RAG]_')

question = st.text_input('Digite uma pergunta paaraa a IA executar uma consulta nos documentos:', "")

if st.button('Enviar'):
    st.write('A pergunta foi: \'', question+'\'')
    url = 'http://127.0.0.1:8000/api'
    paylod = json.dumps({'query': question})
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    response = requests.request('POST', url, headers= headers, data= paylod)
    answer = json.loads(response.text)['answer']
    rege = re.compile('\[Document\ [0-9]+\]|\[[0-9]+\]')
    m = rege.findall(answer)
    num = []

    for n in m:
        num = num + [int(s) for s in re.findall(r'\b\d+\b', n)]
    
    st.markdown(answer)
    documents = json.loads(response.text)['context']
    show_docs = []

    for n in num:
        for doc in documents:
            if int(doc['id']) == n:
                show_docs.append(doc)
    
    var_id = 10

    for doc in show_docs:
        with st.expander(str(doc['id'])+' - '+doc['path']):
            st.write(doc['content'])
            if os.path.exists(doc['path']):
                with open(doc['path'], 'rb') as f:
                    st.download_button('Download do Arquivo', f, file_name=doc['path'].split('/')[-1], key=var_id)
            else:
                st.error(f"Arquivo não encontrado: {doc['path']}")
                var_id = var_id + 1