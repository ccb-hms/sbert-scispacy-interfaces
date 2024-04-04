import os
import spacy
import scispacy
import logging
import argparse
import shortuuid
import truecase
import pandas as pd
from text2term import onto_utils
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import nltk
nltk.download('punkt')

__version__ = "0.4.3"


class ScispacyUmlsNer:

    def __init__(self, model="en_ner_bc5cdr_md"):
        self._log = onto_utils.get_logger("sbert.spelunking", logging.INFO)

        # Load the given scispacy model
        self._model = model
        self._log.info(f"Loading scispaCy model {my_model}...")
        self._ner = spacy.load(self._model)

        # Add abbreviation detector and UMLS linker
        self._ner.add_pipe("abbreviation_detector")
        self._ner.add_pipe("scispacy_linker", config={"resolve_abbreviations": True,
                                                      "linker_name": "umls",
                                                      "filter_for_definitions": False,
                                                      "no_definition_threshold": 0.85,
                                                      "max_entities_per_mention": 1})
        # Headers of output dataframe
        self._output_df_headers = ["ID", "InputText", "Entity", "EntityLabel", "UMLS.CUI", "UMLS.Label",
                                   "UMLS.Definition", "UMLS.SemanticTypeIDs", "UMLS.SemanticTypeLabels",
                                   "UMLS.Synonyms", "UMLS.Score"]

        # Load UMLS Semantic Types table
        # https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
        self._umls_semantic_types = pd.read_csv(
            "https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt",
            sep="|", names=['abbv', 'tui', 'label'])

    def extract_entities(self, text, output_as_df=False):
        temp_entity_id = shortuuid.ShortUUID().random(length=10)
        text = truecase.get_true_case(text)
        output_data = []
        doc = self._ner(text=text)
        if len(doc.ents) == 0:
            self._log.info(f"No named entities found in text: {text}")
            self._add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text, entity="",
                                       entity_label="")
            return output_data
        for entity in doc.ents:  # Extract named entities and link them to UMLS
            linker = self._ner.get_pipe("scispacy_linker")
            if len(entity._.kb_ents) > 0:
                for umls_entity in entity._.kb_ents:
                    cui, score = umls_entity
                    score = round(score, 3)
                    details = linker.kb.cui_to_entity[umls_entity[0]]
                    self._add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text,
                                               entity=entity.text,
                                               entity_label=entity.label_, umls_cui=cui,
                                               umls_label=details.canonical_name,
                                               umls_semantic_types=details.types, umls_definition=details.definition,
                                               umls_synonyms=", ".join(details.aliases), umls_mapping_score=score)
            else:
                self._log.info(f"No UMLS mappings found for entity: {text}")
                self._add_entity_to_output(data=output_data, input_id=temp_entity_id, input_text=text,
                                           entity=entity.text, entity_label=entity.label_)
        if output_as_df:
            return pd.DataFrame(output_data, columns=self._output_df_headers)
        else:
            return output_data

    def extract_entities_in_file(self, text_file):
        self._log.info(f"\nProcessing {text_file}...")
        output_data = []
        with open(text_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                if line != "":
                    output_data.extend(self.extract_entities(line))
        output_df = pd.DataFrame(output_data, columns=self._output_df_headers)
        return output_df

    def _add_entity_to_output(self, data, input_id, input_text, entity, entity_label, umls_cui="", umls_label="",
                              umls_definition="", umls_semantic_types=(), umls_synonyms="", umls_mapping_score=""):
        data.append({"ID": input_id,
                     "InputText": input_text,
                     "Entity": entity,
                     "EntityLabel": entity_label,
                     "UMLS.CUI": umls_cui,
                     "UMLS.Label": umls_label,
                     "UMLS.Definition": umls_definition,
                     "UMLS.SemanticTypeIDs": ",".join(umls_semantic_types),
                     "UMLS.SemanticTypeLabels": self._get_umls_semantic_type_labels(umls_semantic_types),
                     "UMLS.Synonyms": umls_synonyms,
                     "UMLS.Score": umls_mapping_score})

    def _get_umls_semantic_type_labels(self, semantic_types):
        semantic_type_labels = ""
        for semantic_type in semantic_types:
            semantic_type_labels_df = self._umls_semantic_types[self._umls_semantic_types["tui"] == semantic_type]
            semantic_type_labels += semantic_type_labels_df["label"].item() + ","
        return semantic_type_labels.rstrip(",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("scispacy_ner")
    parser.add_argument("-m", "--model", default="en_core_sci_scibert", type=str,
                        help="Name of the scispaCy model to be used.")
    args = parser.parse_args()
    my_model = args.model

    output_dir = os.path.join("output_scispacy", f"output_{my_model}")
    os.makedirs(output_dir, exist_ok=True)

    my_scispacy = ScispacyUmlsNer(my_model)

    df1 = my_scispacy.extract_entities_in_file("data/InfDisease.txt")
    df1.to_csv(os.path.join(output_dir, "InfDisease_scispacy_entities.tsv"), sep="\t", index=False)

    df2 = my_scispacy.extract_entities_in_file("data/InfDfromEFO.txt")
    df2.to_csv(os.path.join(output_dir, "InfDfromEFO_scispacy_entities.tsv"), sep="\t", index=False)
