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

def _get_embeddings(directory_path : str, number : int = None) -> list[DataPoint]:
    '''
    Generate embeddings for OWL files in a directory.

    This method processes `.owl` files in the specified directory and generates embeddings 
    using the OWL2Vec* model. The embeddings are returned as a list of file names and 
    their corresponding graph embeddings.

    Args:
        directory_path (str): Path to the directory containing `.owl` files.
        number (int, optional): Number of files to process. Defaults to `None`, 
            meaning all files in the directory are processed.
        consistency (bool, optional): Currently unused parameter. Reserved for 
            future use.

    Returns:
        list[list[str, np.array]]: A list of pairs where each pair consists of:
            - The file name (str) of the processed `.owl` file.
            - The corresponding graph embedding (np.array).    

    Raises: 
        any Exceptions during processing are logged and skipped
    '''
    filenames = os.listdir(directory_path)
    iterations = number if number else len(filenames)

    rows = {}

    with ThreadPoolExecutor() as executor:
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

def main():
    logging.info("Started Process...")
    logging.info("Embedding Consistent Ontologies...")
    embeddings_consistent : list[DataPoint]= _get_embeddings("../GLaMoR/data/ont_modules/")
    logging.info("Embedding Inconsistent Ontologies...")
    embeddings_inconsistent : list[DataPoint] = _get_embeddings("../GLaMoR/data./ont_modules_inconsistent/")
    logging.info("Finished Embeddings.")
    embeddings = embeddings_consistent + embeddings_inconsistent
    dataset_df : pd.DataFrame = pd.read_csv("../GLaMoR/data/dataset.csv", header=["name", "consistency", "body"])
    embedding_dict : dict[str:np.array] = {name : embedding for [name, embedding] in embeddings}
    embedding_column : list[np.array] = []

    for entry in dataset_df["name"]:
        embedding_column.append(embedding_dict[entry])

    dataset_df["embedding"] = embedding_column

    dataset_df.to_csv("dataset.csv")