# sbert-scispacy-interfaces
prototype interfaces for NER with [scispaCy](https://allenai.github.io/scispacy/) and generating sentence embeddings with [SentenceTransformers (sbert)](https://www.sbert.net)


## scispaCy NER

`scispacy_ner.py` provides the class `ScispacyUmlsNer` that takes as input a scispaCy model name (see models [here](https://allenai.github.io/scispacy/)) and then can extract named entities in:
- a _string_ via the function `extract_entities(<string>)`
- a _file_ containing a list of _strings_ via the function `extract_entities_in_file(<file>)` 

Both functions return a data frame containing:
- `ID` a random UUID assigned to each input string
- `InputText` the input string
- `Entity` an entity detected in the input string 
- `EntityLabel` the "label" of the entity as categorized by scispacy's NER (e.g. `Disease` or `Chemical`)

And then details of the UMLS terms that the detected entities were mapped to in the UMLS Metathesaurus:
- `UMLS.CUI` each term's CUI (UMLS Concept Unique Identifier)
- `UMLS.Label` UMLS preferred label for the term
- `UMLS.Definition` UMLS definition of the term
- `UMLS.SemanticTypeIDs` UMLS (broad) semantic types of the term 
- `UMLS.SemanticTypeLabels` UMLS labels of the semantic types of the term
- `UMLS.Synonyms` UMLS synonyms for the term
- `UMLS.Score` confidence score of the mapping between `Entity` and this UMLS term 