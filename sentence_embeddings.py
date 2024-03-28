import os
import torch
import pickle
import logging
from owlready2 import *
from text2term import OntologyTermCollector, OntologyTermType, onto_utils
from sentence_transformers import SentenceTransformer, util

__version__ = "0.1.1"

LOG = onto_utils.get_logger("sbert.spelunking", logging.INFO)


def get_sentence_embeddings(sentences, sentence_embedder, save_embeddings=True, output_filepath=""):
    """
    Get embeddings for the given sentences using the specified sentence embedder. Optionally save the generated
    embeddings to a given filepath.
    """
    LOG.info("Generating embeddings...")
    sentence_embeddings = sentence_embedder.encode(sentences, convert_to_numpy=True)
    LOG.info("...done")
    if save_embeddings:
        LOG.info("Serializing embeddings...")
        if output_filepath == "":
            output_dir = "output_sbert"
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, "sentence_embeddings.pkl")
        with open(output_filepath, "wb") as output_file:
            pickle.dump({'sentences': sentences, 'embeddings': sentence_embeddings}, output_file)
        LOG.info("...done")
    return sentence_embeddings


def get_ontology_embeddings(ontology_url, sentence_embedder, save_embeddings=True, output_filepath=""):
    """
    Get embeddings for terms in the ontology at the given URL, using the specified sentence embedder.
    Optionally save the generated embeddings to a given filepath.
    """
    terms = _get_ontology_term_sentences(ontology_url=ontology_url)
    term_embeddings = get_sentence_embeddings(sentences=terms, sentence_embedder=sentence_embedder,
                                              save_embeddings=save_embeddings, output_filepath=output_filepath)
    return terms, term_embeddings


def _get_ontology_term_sentences(ontology_url, include_parents=False, include_restrictions=False):
    """
    Get a list of sentences each of which represents an ontology term based on its label, definition, synonyms, IRI,
    and optionally its parents and restrictions of the form 'part-of some Kidney'.
    """
    LOG.info("Preparing ontology term details...")
    term_collector = OntologyTermCollector(ontology_url)
    terms = term_collector.get_ontology_terms(exclude_deprecated=True,
                                              term_type=OntologyTermType.CLASS,
                                              base_iris=(
                                                  "http://www.ebi.ac.uk/efo/", "http://purl.obolibrary.org/obo/MONDO",
                                                  "http://purl.obolibrary.org/obo/HP", "http://www.orpha.net/ORDO",
                                                  "http://purl.obolibrary.org/obo/DOID"))
    output_sentences = []
    for term in terms:
        term_obj = terms[term]
        text_to_embed = ". ".join(str(val) for val in term_obj.labels)

        # add the term definition(s) to the text to embed
        for definition in term_obj.definitions:
            definition = definition.replace("\n", "")
            text_to_embed += f". {definition}"

        # add the term synonym(s) to the text to embed
        for synonym in term_obj.synonyms:
            text_to_embed += f". {term_obj.label} is also known as {synonym}"

        # add the parents
        if include_parents:
            for parent in term_obj.parents.values():
                text_to_embed += f". {term_obj.label} is a kind of {parent}"

        if include_restrictions:
            restrictions = term_obj.restrictions
            for restriction in restrictions:
                text_to_embed += f". {term_obj.label} {restriction} {restrictions[restriction]}"

        # append the term IRI
        text_to_embed += f". IRI<{term_obj.iri}>"
        text_to_embed = text_to_embed.replace("..", ".")
        output_sentences.append(text_to_embed)
    LOG.info("...done")
    return output_sentences


def topk_most_similar_cosine(queries, sentence_embedder, sentence_embeddings, k=5):
    """
    Print the closest k sentences in the corpus for each query sentence based on cosine similarity.
    """
    top_k = min(k, len(corpus))
    for query in queries:
        query_embedding = sentence_embedder.encode(query, convert_to_numpy=True)

        # use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("======================")
        print("Query:", query)
        print("\nTop-k most similar sentences in corpus (cosine similarity):")

        for score, idx in zip(top_results[0], top_results[1]):
            print("-", corpus[idx], "(Score: {:.4f})".format(score))


def topk_most_similar_semantic_search(queries, sentence_embedder, sentence_embeddings, k=5):
    """
    Print the closest k sentences in the corpus for each query sentence based on semantic search.
    """
    top_k = min(k, len(corpus))
    for query in queries:
        query_embedding = sentence_embedder.encode(query, convert_to_numpy=True)

        hits = util.semantic_search(query_embedding, sentence_embeddings, top_k=top_k)
        hits = hits[0]  # Get the hits for the first query

        print("======================")
        print("Query:", query)
        print("\nTop-k most similar sentences in corpus (semantic search):")
        for hit in hits:
            print("-", corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))


if __name__ == '__main__':
    efo_url = "https://github.com/EBISPOT/efo/releases/download/v3.64.0/efo.owl"
    output_dir = "output_sbert"
    os.makedirs(output_dir, exist_ok=True)
    embeddings_output_file = os.path.join(output_dir, "efo_embeddings.pkl")

    # create sentence embedder based on the specified model
    # TODO try also: MSMARCO roberta-v3  allenai-specter
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # get embeddings for the ontology terms, along with the corpus of ontology term "sentences"
    corpus, embeddings = get_ontology_embeddings(sentence_embedder=embedder, ontology_url=efo_url,
                                                 save_embeddings=True, output_filepath=embeddings_output_file)

    # example queries
    myqueries = [
        "Myocardial Infarction",
        "heart attack",
        "alzeimer disease, alzeimer disease",
        "physical activity",
        "kuru"
    ]

    # do some top-k most similar searches
    topk_most_similar_cosine(queries=myqueries, sentence_embedder=embedder, sentence_embeddings=embeddings, k=5)
    topk_most_similar_semantic_search(queries=myqueries, sentence_embedder=embedder, sentence_embeddings=embeddings)
