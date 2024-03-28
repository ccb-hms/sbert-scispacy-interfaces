import os
import spacy
import scispacy
import shortuuid
import truecase
import pandas as pd
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import nltk
nltk.download('punkt')

__version__ = "0.3.1"

# Load the scispacy model and add UMLS linker and abbreviation detector
MODEL = "en_ner_bc5cdr_md"  # "en_core_sci_scibert"
NLP = spacy.load(MODEL)
NLP.add_pipe("abbreviation_detector")
NLP.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                                        "linker_name": "umls",
                                        "filter_for_definitions": False,
                                        "no_definition_threshold": 0.85,
                                        "max_entities_per_mention": 1})

# Load UMLS Semantic Types table (https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html)
UMLS_SEMANTIC_TYPES = pd.read_csv("https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt",
                                  sep="|", names=['abbv', 'tui', 'label'])


def extract_entities(text):
    temp_entity_id = shortuuid.ShortUUID().random(length=10)
    text = truecase.get_true_case(text)
    output_data = []
    doc = NLP(text=text)
    if len(doc.ents) == 0:
        print(f"No named entities found in text: {text}")
        _add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text, entity="", entity_label="")
        return output_data
    for entity in doc.ents:  # Extract named entities and link them to UMLS
        linker = NLP.get_pipe("scispacy_linker")
        if len(entity._.kb_ents) > 0:
            for umls_entity in entity._.kb_ents:
                cui, score = umls_entity
                score = round(score, 3)
                details = linker.kb.cui_to_entity[umls_entity[0]]
                _add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text, entity=entity.text,
                                      entity_label=entity.label_, umls_cui=cui, umls_label=details.canonical_name,
                                      umls_semantic_types=details.types, umls_definition=details.definition,
                                      umls_synonyms=", ".join(details.aliases), umls_mapping_score=score)
        else:
            print(f"No UMLS mappings found for entity: {text}")
            _add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text, entity=entity.text,
                                  entity_label=entity.label_)
    return output_data


def _add_entity_to_output(data, input_id, input_text, entity, entity_label, umls_cui="", umls_label="",
                          umls_definition="", umls_semantic_types=(), umls_synonyms="", umls_mapping_score=""):
    data.append({"ID": input_id,
                 "InputText": input_text,
                 "Entity": entity,
                 "EntityLabel": entity_label,
                 "UMLS.CUI": umls_cui,
                 "UMLS.Label": umls_label,
                 "UMLS.Definition": umls_definition,
                 "UMLS.SemanticTypeIDs": ",".join(umls_semantic_types),
                 "UMLS.SemanticTypeLabels": _get_umls_semantic_type_labels(umls_semantic_types),
                 "UMLS.Synonyms": umls_synonyms,
                 "UMLS.Score": umls_mapping_score})


def _get_umls_semantic_type_labels(semantic_types):
    semantic_type_labels = ""
    for semantic_type in semantic_types:
        semantic_type_labels_df = UMLS_SEMANTIC_TYPES[UMLS_SEMANTIC_TYPES["tui"] == semantic_type]
        semantic_type_labels += semantic_type_labels_df["label"].item() + ","
    return semantic_type_labels.rstrip(",")


def extract_entities_in_file(text_file):
    print(f"\nProcessing {text_file}...")
    output_data = []
    with open(text_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.replace("\n", "")
            if line != "":
                output_data.extend(extract_entities(line))
    output_df = pd.DataFrame(output_data, columns=["ID", "InputText", "Entity", "EntityLabel", "UMLS.CUI", "UMLS.Label",
                                                   "UMLS.Definition", "UMLS.SemanticTypeIDs", "UMLS.SemanticTypeLabels",
                                                   "UMLS.Synonyms", "UMLS.Score"])
    return output_df


if __name__ == '__main__':
    print(f"Extracting entities using scispacy model {MODEL}...")
    output_dir = os.path.join("output_scispacy", f"output_{MODEL}")
    os.makedirs(output_dir, exist_ok=True)

    df1 = extract_entities_in_file("data/InfDisease.txt")
    df1.to_csv(os.path.join(output_dir, "InfDisease_scispacy_entities.tsv"), sep="\t", index=False)

    df2 = extract_entities_in_file("data/InfDfromEFO.txt")
    df2.to_csv(os.path.join(output_dir, "InfDfromEFO_scispacy_entities.tsv"), sep="\t", index=False)
