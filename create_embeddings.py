from owl2vec_star import owl2vec_star
import os
import numpy as np
from tqdm import tqdm
from random import shuffle
from owlready2 import get_ontology
import traceback
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

PREFIXES : List[str] = ["AIO","EID","OIL","OILWI","OILWPI","UE","UEWI1","UEWI2","UEWPI","UEWIP","SOSINETO","CSC","OOR","OOD"]

type DataPoint = list[str, np.array]
type Embedding = np.array
#%%
def _create_graph_embedding(embeddings):
    ''' 
    Given the KeyedVector Embeddings of all words in an ontology, 
    returns the average of the vectors as graph embeddings
    '''
    words = embeddings.key_to_index
    vectors = [embeddings[word] for word in words]
    graph_embedding = np.mean(vectors, axis=0)
    return graph_embedding
# %%

def process_file(filename):
    """Processes a single file to extract its graph embedding."""
    directory_path = "../GLaMoR/data./ont_modules_inconsistent/" if filename.split("_")[0] in PREFIXES else "../GLaMoR/data/ont_modules/"
    try:
        gensim_model = owl2vec_star.extract_owl2vec_model(
            f"{directory_path}{filename}", "./default.cfg", True, True, True
        )
        graph_embedding = _create_graph_embedding(gensim_model.wv)
        return filename, graph_embedding
    except Exception as exception:
        traceback.print_exc()
        return None

def _get_embeddings(file_names : List[int], number: int = None, workers: int = None) -> dict[str, Embedding]:
    """
    Generate embeddings for OWL files in a directory using multithreading.

    Args:
        directory_path (str): Path to the directory containing `.owl` files.
        number (int, optional): Number of files to process. Defaults to `None`, meaning all files.
        workers (int, optional): Number of threads to use for parallel processing. Defaults to `os.cpu_count()`.

    Returns:
        dict: A dictionary with file names as keys and their graph embeddings as values.
    """
    iterations = number if number else len(file_names)
    max_workers = workers if workers else os.cpu_count()

    embeddings: Dict[str, np.ndarray] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, file): file for file in file_names[:iterations]}
        for future in tqdm(futures.keys(), total=iterations):
            result = future.result()
            if result:
                filename, graph_embedding = result
                embeddings[filename] = graph_embedding

    return embeddings

def _add_embedding_to_dataframe(dataframe : pd.DataFrame, embeddings: dict[str, Embedding]) -> pd.DataFrame:
    default_embedding = np.zeros(next(iter(embeddings.values())).shape)
    dataframe["embedding"] = dataframe["file_name"].apply(
        lambda x: embeddings.get(x, default_embedding)
    )
    return dataframe

def main():

    MAX_LENGTH = 4096
    df = pd.read_csv("../dataset.csv", header=0)

    subset_df = df.loc(df["tokenized_length"] < MAX_LENGTH)

    file_names = subset_df["file_name"].values
    
    print(file_names)

    logging.info("Started Process...")
    logging.info("Embedding Ontologies Ontologies...")
    embeddings : dict[str, Embedding]= _get_embeddings("")
    logging.info("Finished Embeddings...")
    logging.info("Building DataFrame...")
    dataset_df = _add_embedding_to_dataframe(subset_df, embeddings)
    dataset_df.to_csv("dataset.csv")

if __name__ == "__main__":
    main()