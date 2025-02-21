import datetime
import chromadb
import traceback
import pandas as pd
import json
from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def create_database(collection):
    file_name = 'COA_OpenData.csv'
    df = pd.read_csv(file_name)
    print(df.columns)

    for index, row in df.iterrows():
        metadata = {
            'file_name': file_name,
            'name':row['Name'],
            'type':row['Type'],
            'address':row['Address'],
            'tel':row['Tel'],
            'city':row['City'],
            'town':row['Town'],
            'date':(int)(datetime.datetime.strptime(row['CreateDate'], "%Y-%m-%d").timestamp())}
        #print(metadata)
        collection.add(
            documents=[row['HostWords']],
            metadatas=[metadata],
            ids=['{}'.format(index)]
        )

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    #create_database(collection)
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    #formatted_output = json.dumps(results, indent=4, ensure_ascii=False)
    #print('results={}'.format(formatted_output))

    filtered_results = []
    for index, distance in enumerate(results['distances'][0]):
            if 1-distance > 0.8:
                name = results['metadatas'][0][index]['name']
                filtered_results.append(name)

    print(filtered_results)
    return filtered_results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()

    results = collection.query(
        query_texts=['查询名称'],
        where={"name": store_name}
    )
    #print('results={}'.format(results))

    for index, ids in enumerate(results['ids'][0]):
         metadatas = results['metadatas'][0][index]
         metadatas['new_store_name'] = new_store_name
         collection.update([ids], metadatas=[metadatas])

    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    # formatted_output = json.dumps(results, indent=4, ensure_ascii=False)
    # print('results={}'.format(formatted_output))

    filtered_results = []
    for index, distance in enumerate(results['distances'][0]):
            if 1-distance > 0.8:
                new_store_name = results['metadatas'][0][index].get('new_store_name', "")
                name = results['metadatas'][0][index]['name']
                filtered_results.append(new_store_name if new_store_name else name)

    print(filtered_results)
    return filtered_results
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
