# sbert-scispacy-tools
prototype tools for NER with [scispaCy](https://allenai.github.io/scispacy/) and generating sentence embeddings with [SentenceTransformers (sbert)](https://www.sbert.net)

## scispaCy NER
`scispacy_ner.py` provides the class `ScispacyUmlsNer` that takes as input a scispaCy model name (see models [here](https://allenai.github.io/scispacy/)) and then can extract named entities in a:
- _string_ using the function `extract_entities(<string>)`
- _list of strings_ using the function `extract_entities_in_list(<string_list>)`
- _file_ containing a list of _strings_ using the function `extract_entities_in_file(<filepath>)`
- _table_ containing a column of _strings_ and (optionally) associated _identifiers_ using the function `extract_entities_in_table(<filepath> <input_text_col> [<input_id_col>])`

Both functions return a data frame containing:
- `InputID` a random UUID assigned to each input string
- `InputText` the input string
- `Entity` an entity detected in the input string 
- `EntityType` the type of entity as categorized by scispacy's NER (e.g. `Disease` or `Chemical`)

And then details of the UMLS terms that the detected entities were mapped to in the UMLS Metathesaurus:
- `UMLS.CUI` each term's CUI (UMLS Concept Unique Identifier)
- `UMLS.Label` UMLS preferred label for the term
- `UMLS.Definition` UMLS definition of the term
- `UMLS.Synonyms` UMLS synonyms for the term
- `UMLS.SemanticTypeIDs` UMLS (broad) semantic types of the term 
- `UMLS.SemanticTypeLabels` UMLS labels of the semantic types of the term
- `UMLS.MappingScore` confidence score of the mapping between `Entity` and this UMLS term 

### Example Usage

Instantiate `ScispacyUmlsNer` with a model of interest, e.g. _en_core_sci_scibert_:
```python
myspacy = ScispacyUmlsNer(model="en_core_sci_scibert")
```

Extract entities in a string and get a list of entities of type `LinkedNamedEntity`:

```python
ents = myspacy.extract_entities(input_text="my dog Milo has the flu")
```

Extract entities in a file and get a data frame:
```python
df = myspacy.extract_entities_in_file(filepath="example/file.txt", output_as_df=True)
```

Extract entities in a table and get a data frame:
```python
df = myspacy.extract_entities_in_table(filepath="mytable.tsv", output_as_df=True,
                                       input_text_col="Input", input_id_col="InputID")
```

It is also possible to use this module from a terminal, for example: 
```shell
python scispacy_ner.py --input data/gwascatalog_metadata.tsv --model en_ner_bc5cdr_md --col "DISEASE.TRAIT" --id "STUDY.ACCESSION"
```

### Models

| Model	                                                                                                                        | Entity Types                                                                                                                                                                                                                                                                             |
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [en_ner_craft_md](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz)	           | GGP, SO, TAXON, CHEBI, GO, CL                                                                                                                                                                                                                                                            |
| [en_ner_jnlpba_md](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz)	         | DNA, CELL_TYPE, CELL_LINE, RNA, PROTEIN                                                                                                                                                                                                                                                  |
| [en_ner_bc5cdr_md](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz)	         | DISEASE, CHEMICAL                                                                                                                                                                                                                                                                        |
| [en_ner_bionlp13cg_md](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz)	 | AMINO_ACID, ANATOMICAL_SYSTEM, CANCER, CELL, CELLULAR_COMPONENT, ORGAN, TISSUE, ORGANISM, DEVELOPING_ANATOMICAL_STRUCTURE, GENE_OR_GENE_PRODUCT, IMMATERIAL_ANATOMICAL_ENTITY, MULTI-TISSUE_STRUCTURE, ORGANISM_SUBDIVISION, ORGANISM_SUBSTANCE, PATHOLOGICAL_FORMATION, SIMPLE_CHEMICAL |
| [en_core_sci_scibert](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz)    | A full spaCy pipeline for biomedical data with a ~785k vocabulary and allenai/scibert-base as the transformer model                                                                                                                                                                      |

Models must be individually installed as such:

```python
pip install <Model URL>
```
for example:
```python
pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz"
```


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