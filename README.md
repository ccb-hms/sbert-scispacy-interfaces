# sbert-scispacy-interfaces
prototype tools for NER with [scispaCy](https://allenai.github.io/scispacy/) and generating sentence embeddings with [SentenceTransformers (sbert)](https://www.sbert.net)

## scispaCy NER
`scispacy_ner.py` provides the class `ScispacyUmlsNer` that takes as input a scispaCy model name (see models [here](https://allenai.github.io/scispacy/)) and then can extract named entities in:
- a _string_ via the function `extract_entities(<string>)`
- a _list of strings_ via the function `extract_entities_in_list(<string_list>)`
- a _file_ containing a list of _strings_ via the function `extract_entities_in_file(<file>)` 

Both functions return a data frame containing:
- `InputID` a random UUID assigned to each input string
- `InputText` the input string
- `Entity` an entity detected in the input string 
- `EntityType` the type of entity as categorized by scispacy's NER (e.g. `Disease` or `Chemical`)

And then details of the UMLS terms that the detected entities were mapped to in the UMLS Metathesaurus:
- `UMLS.CUI` each term's CUI (UMLS Concept Unique Identifier)
- `UMLS.Label` UMLS preferred label for the term
- `UMLS.Definition` UMLS definition of the term
- `UMLS.SemanticTypeIDs` UMLS (broad) semantic types of the term 
- `UMLS.SemanticTypeLabels` UMLS labels of the semantic types of the term
- `UMLS.Synonyms` UMLS synonyms for the term
- `UMLS.Score` confidence score of the mapping between `Entity` and this UMLS term 


## Sentence-BERT (sbert) embeddings
`sbert_embedder.py` provides the class `SbertEmbedder` that takes as input the name of a sentence embedding model (see models [here](https://www.sbert.net/docs/pretrained_models.html)), and then can compare two lists of strings (or two files containing lists of strings), or a list of strings with an ontology, based on embeddings generated for those strings using the specified embedding model.

### Initializing
To start, create a sentence embedder using a model of choice (by default `all-MiniLM-L6-v2`). 
```python
SbertEmbedder(embedding_model="all-MiniLM-L6-v2")
```

### Generating embeddings
There are two functions to create embeddings:

```python
get_sentence_embeddings(sentences, save_embeddings=False, output_filepath="")
```
generates embeddings for input sentences and optionally serializes the embeddings to the specified filepath.

```python
get_ontology_embeddings(ontology_url, save_embeddings=False, output_filepath="")
```
generates embeddings for terms in the ontology at the given URL, and optionally serializes the embeddings to
        the specified filepath.

### Comparing embeddings
There are three functions to perform cosine similarity-based comparisons of embeddings:
```python
compare_sentences(left_sentences, right_sentences, top_k=3)
```
compares two lists of sentences and returns a data frame containing, for each sentence in `left_sentences` up to _k_ `top_k` closest sentences from `right_sentences`.

```python
compare_files(left_file, right_file, top_k=3)
```
a convenience function to compare two files each containing a list of sentences.

```python
compare_to_ontology(queries, ontology_url, top_k=3)
```
compares a list of sentences against embeddings of terms in the specified ontology. Returns a data frame containing, for each sentence in `query_sentences`, up to _k_ `top_k` closest ontology terms.