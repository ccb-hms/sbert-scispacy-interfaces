import os
import spacy
import scispacy
import logging
import argparse
import shortuuid
import truecase
import pandas as pd
from tqdm import tqdm
from text2term import onto_utils
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from named_entity import LinkedNamedEntity
import nltk

nltk.download('punkt')

__version__ = "0.6.3"


class ScispacyUmlsNer:

    def __init__(self, model="en_ner_bc5cdr_md"):
        self._log = onto_utils.get_logger("scispacy.ner", logging.INFO)

        # Load the given scispacy model
        self._model = model
        self._log.info(f"Loading scispaCy model {model}...")
        self._ner = spacy.load(self._model)

        # Add abbreviation detector and UMLS linker
        self._ner.add_pipe("abbreviation_detector")
        self._ner.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                                                      "linker_name": "umls",
                                                      "filter_for_definitions": False,
                                                      "no_definition_threshold": 0.85,
                                                      "max_entities_per_mention": 1})

        # Load UMLS Semantic Types table
        # https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
        self._umls_semantic_types = pd.read_csv(
            "https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt",
            sep="|", names=['abbv', 'tui', 'label'])

    def extract_entities(self, text, output_as_df=False, incl_unlinked_entities=False):
        temp_entity_id = shortuuid.ShortUUID().random(length=10)
        truecased_text = truecase.get_true_case(text)
        entities = []
        doc = self._ner(text=truecased_text)
        if len(doc.ents) == 0:
            self._log.debug(f"No named entities found in text: {text}")
        for entity in doc.ents:  # Extract named entities and link them to UMLS
            linker = self._ner.get_pipe("scispacy_linker")
            if len(entity._.kb_ents) > 0:
                for umls_entity in entity._.kb_ents:
                    cui, score = umls_entity
                    score = round(score, 3)
                    details = linker.kb.cui_to_entity[umls_entity[0]]
                    self._add_entity_to_output(output=entities, input_id=temp_entity_id, input_text=text,
                                               entity=entity.text, entity_type=entity.label_, umls_cui=cui,
                                               umls_label=details.canonical_name, umls_semantic_types=details.types,
                                               umls_definition=details.definition,
                                               umls_synonyms=", ".join(details.aliases), umls_mapping_score=score)
            else:
                self._log.debug(f"No UMLS mappings found for entity: {text}")
                if incl_unlinked_entities:
                    self._add_entity_to_output(output=entities, input_id=temp_entity_id, input_text=text,
                                               entity=entity.text, entity_type=entity.label_)
        if output_as_df:
            return pd.DataFrame(entities)
        else:
            return entities

    def extract_entities_in_list(self, string_list, output_as_df=False, incl_unlinked_entities=False):
        entities = []
        for string in tqdm(string_list):
            entities.extend(self.extract_entities(string, incl_unlinked_entities=incl_unlinked_entities))
        if output_as_df:
            return pd.DataFrame(entities)
        else:
            return entities

    def extract_entities_in_file(self, filepath, output_as_df=False, incl_unlinked_entities=False):
        self._log.info(f"Processing {filepath}...")
        entities = []
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line.replace("\n", "")
                if line != "":
                    entities.extend(self.extract_entities(line, incl_unlinked_entities=incl_unlinked_entities))
        if output_as_df:
            return pd.DataFrame(entities)
        else:
            return entities

    def _add_entity_to_output(self, output, input_id, input_text, entity, entity_type, umls_cui="", umls_label="",
                              umls_definition="", umls_semantic_types=(), umls_synonyms="", umls_mapping_score=""):
        entity = LinkedNamedEntity(input_id=input_id, input_text=input_text, entity=entity, entity_type=entity_type,
                                   umls_cui=umls_cui, umls_label=umls_label, umls_definition=umls_definition,
                                   umls_synonyms=umls_synonyms, umls_semantic_type_ids=",".join(umls_semantic_types),
                                   umls_semantic_type_labels=self._get_umls_semantic_type_labels(umls_semantic_types),
                                   umls_mapping_score=umls_mapping_score)
        output.append(entity.as_dict())

    def _get_umls_semantic_type_labels(self, semantic_types):
        semantic_type_labels = ""
        for semantic_type in semantic_types:
            semantic_type_labels_df = self._umls_semantic_types[self._umls_semantic_types["tui"] == semantic_type]
            semantic_type_labels += semantic_type_labels_df["label"].item() + ","
        return semantic_type_labels.rstrip(",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("scispacy_ner")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input file")
    parser.add_argument("-m", "--model", default="en_core_sci_scibert", type=str,
                        help="Name of the scispaCy model to be used")
    args = parser.parse_args()
    input_model = args.model
    input_file = args.input

    # prepare output folder and file
    output_dir = os.path.join("", "output", "scispacy_ner", f"model_{input_model}")
    output_file_path = os.path.join(output_dir, input_file.split(os.sep)[-1] + "_entities.tsv")
    os.makedirs(output_dir, exist_ok=True)

    # instantiate scispacy with the specified model
    my_scispacy = ScispacyUmlsNer(model=input_model)

    # extract entities in the given input file
    entities_df = my_scispacy.extract_entities_in_file(filepath=input_file, output_as_df=True)
    entities_df.to_csv(output_file_path, sep="\t", index=False)
