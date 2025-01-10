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

def process_file(directory_path, filename):
    """Processes a single file to extract its graph embedding."""
    try:
        gensim_model = owl2vec_star.extract_owl2vec_model(
            f"{directory_path}{filename}", "./default.cfg", True, True, True
        )
        graph_embedding = _create_graph_embedding(gensim_model.wv)
        return filename, graph_embedding
    except Exception as exception:
        traceback.print_exc()
        return None

def _get_embeddings(directory_path: str, number: int = None, workers: int = None) -> dict[str, Embedding]:
    """
    Generate embeddings for OWL files in a directory using multithreading.

    Args:
        directory_path (str): Path to the directory containing `.owl` files.
        number (int, optional): Number of files to process. Defaults to `None`, meaning all files.
        workers (int, optional): Number of threads to use for parallel processing. Defaults to `os.cpu_count()`.

    Returns:
        dict: A dictionary with file names as keys and their graph embeddings as values.
    """
    filenames = os.listdir(directory_path)
    iterations = number if number else len(filenames)
    max_workers = workers if workers else os.cpu_count()

    rows = {}


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for each file
        futures = [
            executor.submit(process_file, directory_path, filenames[i])
            for i in range(iterations)
        ]

        for future in tqdm(as_completed(futures), total=iterations):
            result = future.result()
            if result:
                filename, graph_embedding = result
                rows[filename] = graph_embedding

    return rows

def _add_embedding_to_dataframe(dataframe_path: str, embeddings: dict[str, Embedding]) -> pd.DataFrame:
    dataset_df: pd.DataFrame = pd.read_csv(dataframe_path, names=["name", "consistency", "body"], header=0)
    embedding_column : list[np.array] = []
    
    for entry in dataset_df["name"]:
        embedding_column.append(embeddings.get(entry, np.zeros_like(next(iter(embeddings.values())))))


    dataset_df["embedding"] = embedding_column

    return dataset_df

def main():
    logging.info("Started Process...")
    logging.info("Embedding Consistent Ontologies...")
    embeddings_consistent : dict[str, Embedding]= _get_embeddings("../GLaMoR/data/ont_modules/")
    logging.info("Embedding Inconsistent Ontologies...")
    embeddings_inconsistent : dict[str, Embedding] = _get_embeddings("../GLaMoR/data./ont_modules_inconsistent/")
    logging.info("Finished Embeddings...")
    embeddings = {**embeddings_consistent, **embeddings_inconsistent}

    logging.info("Building DataFrame...")
    dataset_df = _add_embedding_to_dataframe("../GLaMoR/data/dataset.csv", embeddings)
    dataset_df.to_csv("dataset.csv")