import torch
import pickle
import logging
import pandas as pd
from owlready2 import *
from text2term import OntologyTermCollector, OntologyTermType, onto_utils
from sentence_transformers import SentenceTransformer, util

__version__ = "0.2.2"


class SbertEmbedder:

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self._embedding_model = embedding_model

        # create sentence embedder based on the specified embedding model
        self._sentence_embedder = SentenceTransformer(self._embedding_model)
        self._log = onto_utils.get_logger("sbert.embedder", logging.INFO)

    def get_sentence_embeddings(self, sentences, description="sentence", save_embeddings=False, output_filepath=""):
        """
        Get embeddings for the given sentences. Optionally serialize the generated embeddings to the specified filepath.
        """
        self._log.info(f"Generating {description} embeddings using model {self._embedding_model}...")
        sentence_embeddings = self._sentence_embedder.encode(sentences, convert_to_numpy=True)
        self._log.info("...done")
        if save_embeddings:
            self._log.info("Serializing embeddings...")
            if output_filepath == "":
                output_dir = "output_sbert"
                os.makedirs(output_dir, exist_ok=True)
                output_filepath = os.path.join(output_dir, "sentence_embeddings.pkl")
            with open(output_filepath, "wb") as output_file:
                pickle.dump({'sentences': sentences, 'embeddings': sentence_embeddings}, output_file)
            self._log.info("...done")
        return sentence_embeddings

    def get_ontology_embeddings(self, ontology_url, description="ontology", save_embeddings=False, output_filepath="",
                                include_definition=False, include_parents=False, include_restrictions=False):
        """
        Get embeddings for terms in the ontology at the given URL. Optionally serialize the generated embeddings to
        the specified filepath.
        """
        term_sentences = self._get_ontology_term_sentences(ontology_url=ontology_url,
                                                           include_definition=include_definition,
                                                           include_parents=include_parents,
                                                           include_restrictions=include_restrictions)
        term_embeddings = self.get_sentence_embeddings(sentences=term_sentences,
                                                       description=description,
                                                       save_embeddings=save_embeddings,
                                                       output_filepath=output_filepath)
        return term_sentences, term_embeddings

    def _get_ontology_term_sentences(self, ontology_url, include_definition, include_parents, include_restrictions):
        """
        Get a list of sentences each of which represents an ontology term based on its label, synonyms, IRI,
        and optionally its definition, parents, and restrictions of the form 'part-of some Kidney'.
        """
        self._log.info("Preparing ontology term details...")
        term_collector = OntologyTermCollector(ontology_url)
        terms = term_collector.get_ontology_terms(exclude_deprecated=True,
                                                  term_type=OntologyTermType.CLASS,
                                                  base_iris=(
                                                      "http://www.ebi.ac.uk/efo/",
                                                      "http://purl.obolibrary.org/obo/MONDO",
                                                      "http://purl.obolibrary.org/obo/HP", "http://www.orpha.net/ORDO",
                                                      "http://purl.obolibrary.org/obo/DOID"))
        term_sentences = []
        for term in terms:
            term_obj = terms[term]
            text_to_embed = ". ".join(str(val) for val in term_obj.labels)

            # add the term synonym(s) to the text to embed
            for synonym in term_obj.synonyms:
                text_to_embed += f". {term_obj.label} is also known as {synonym}"

            # add the term definition(s) to the text to embed
            if include_definition:
                for definition in term_obj.definitions:
                    definition = definition.replace("\n", "")
                    text_to_embed += f". {definition}"

            # add the parents of the term to the text to embed
            if include_parents:
                for parent in term_obj.parents.values():
                    text_to_embed += f". {term_obj.label} is a kind of {parent}"

            # add any other logical restrictions of the term to the text to embed
            if include_restrictions:
                restrictions = term_obj.restrictions
                for restriction in restrictions:
                    text_to_embed += f". {term_obj.label} {restriction} {restrictions[restriction]}"

            # append the term IRI to the text to embed, so that we can tell what ontology term a sentence represents
            text_to_embed += f". IRI<{term_obj.iri}>"
            text_to_embed = text_to_embed.replace("..", ".")
            term_sentences.append(text_to_embed)
        self._log.info("...done")
        return term_sentences

    def compare_sentences(self, left_sentences, right_sentences, left_embeddings=None, right_embeddings=None, top_k=3,
                          left_description="left", right_description="right"):
        """
        Compare two lists of sentences. Returns a data frame containing:
            for each sentence in `left_sentences', up to k (top_k) closest sentences in 'right_sentences'.
        """
        if left_embeddings is None:
            left_embeddings = self.get_sentence_embeddings(left_sentences, description=left_description)
        if right_embeddings is None:
            right_embeddings = self.get_sentence_embeddings(right_sentences, description=right_description)

        # get cosine similarity scores between sentence embedding lists
        cosine_scores = util.cos_sim(left_embeddings, right_embeddings)

        # find the top-k highest scores and their indices
        top_k_values, top_k_indices = torch.topk(cosine_scores, k=top_k)

        # build data frame containing the top-k pairs of sentences from left and right lists
        results = []
        for i, (values, indices) in enumerate(zip(top_k_values, top_k_indices)):
            for value, index in zip(values, indices):
                results.append({
                    left_description: left_sentences[i],
                    right_description: right_sentences[index],
                    'score': value.item()
                })
        return pd.DataFrame(results)

    def compare_files(self, left_file, right_file, top_k=3, left_description="left", right_description="right"):
        """
        Load and compare two files containing lists of sentences. Returns a data frame containing:
            for each sentence in `left_sentences', up to k (top_k) closest sentences in 'right_sentences'.
        """
        left_sentences = self._load_file(left_file)
        right_sentences = self._load_file(right_file)
        return self.compare_sentences(left_sentences=left_sentences, right_sentences=right_sentences, top_k=top_k,
                                      left_description=left_description, right_description=right_description)

    def compare_to_ontology(self, queries, ontology_url, top_k=3, queries_description="query",
                            ontology_description="ontology", save_embeddings=False, output_filepath="",
                            include_definition=False, include_parents=False, include_restrictions=False):
        """
        Compare a list of sentences with embeddings of terms in the specified ontology. Returns a data frame containing:
            for each sentence in `query_sentences', up to k (top_k) closest ontology terms.
        """
        # build a corpus of sentences representing ontology terms, and then generate embeddings for those sentences
        term_sentences, term_embeddings = self.get_ontology_embeddings(ontology_url=ontology_url,
                                                                       description=ontology_description,
                                                                       save_embeddings=save_embeddings,
                                                                       output_filepath=output_filepath,
                                                                       include_definition=include_definition,
                                                                       include_parents=include_parents,
                                                                       include_restrictions=include_restrictions)
        return self.compare_sentences(queries, term_sentences, None, term_embeddings, top_k=top_k,
                                      left_description=queries_description, right_description=ontology_description)

    @staticmethod
    def _load_file(filepath):
        """
        Load a file containing a newline-delimited list of sentences
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()
            sentences = [line.strip() for line in lines if line.strip() != ""]
            return sentences

    @staticmethod
    def load_embeddings_from_pickle(filepath):
        """
        Load a pickle file containing a dictionary of sentences and their embeddings
        """
        with open(filepath, "rb") as input_pickle:
            pickled_data = pickle.load(input_pickle)
            corpus_sentences = pickled_data["sentences"]
            corpus_embeddings = pickled_data["embeddings"]
            return corpus_sentences, corpus_embeddings


if __name__ == '__main__':
    output_folder = "output_sbert"
    os.makedirs(output_folder, exist_ok=True)

    # create sentence embedder using the specified model
    my_sbert = SbertEmbedder(embedding_model="all-MiniLM-L6-v2")

    # example queries
    my_queries = ["Myocardial Infarction", "heart attack", "alzeimer disease, alzeimer disease", "alzheimer's",
                  "physical activity", "exercising", "lyme", "lime disease", "kuru"]

    """
    EXAMPLE 1: COMPARE TWO LISTS OF SENTENCES
    """
    my_other_queries = ["heart failure", "heart", "heart part", "alzheimer's disease", "exercise",
                        "lyme disease", "Kuru"]
    result = my_sbert.compare_sentences(my_queries, my_other_queries, top_k=3)
    result.to_csv(os.path.join(output_folder, "sentences_vs_sentences.tsv"), sep="\t", index=False)

    """
    EXAMPLE 2: COMPARE A LIST OF STRINGS TO AN ONTOLOGY
    """
    efo_url = "https://github.com/EBISPOT/efo/releases/download/v3.64.0/efo.owl"
    embeddings_output_file = os.path.join(output_folder, "efo_embeddings.pkl")

    # compare the query string list to embeddings of EFO terms, and save the ontology embeddings to a file
    results_efo = my_sbert.compare_to_ontology(my_queries, ontology_url=efo_url, ontology_description="EFO ontology",
                                               top_k=3, save_embeddings=True, output_filepath=embeddings_output_file)
    results_efo.to_csv(os.path.join(output_folder, "sentences_vs_efo.tsv"), sep="\t", index=False)

    """
    EXAMPLE 3: LOAD (ONTOLOGY) EMBEDDINGS FILE FROM DISK
    """
    efo_sentences, efo_embeddings = my_sbert.load_embeddings_from_pickle(embeddings_output_file)
    print(f"Sentence: {efo_sentences[0]}\n")
    print(f"Embedding: {efo_embeddings[0]}")

    """
    EXAMPLE 4: COMPARE TWO FILES CONTAINING LISTS OF SENTENCES
    """
    results_files = my_sbert.compare_files(left_file="data/InfDisease.txt", right_file="data/InfDfromEFO.txt", top_k=2,
                                           left_description="InfDisease", right_description="InfDfromEFO")
    results_files.to_csv(os.path.join(output_folder, "InfDisease_vs_InfDfromEFO.tsv"), sep="\t", index=False)
