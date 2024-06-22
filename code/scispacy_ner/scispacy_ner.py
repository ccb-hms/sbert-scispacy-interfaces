"""Provides ScispacyUmlsNer class"""

import os
import re
import sys
import spacy
import scispacy
import logging
import argparse
import shortuuid
import truecase
import pandas as pd
from tqdm import tqdm
from scispacy.linking import EntityLinker
from named_entity import LinkedNamedEntity
import warnings
import nltk

nltk.download('punkt')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

__version__ = "0.8.0"


class ScispacyUmlsNer:

    def __init__(self, model="en_core_sci_scibert"):
        self._log = ScispacyUmlsNer.get_logger("scispacy.ner", logging.INFO)

        # Load the given scispacy model
        self._model = model
        self._log.info(f"Loading scispaCy model {model}...")
        self._ner = spacy.load(self._model)
        self._log.info("...done")

        # Add UMLS linking pipe
        self._ner.add_pipe("scispacy_linker",
                           config={"resolve_abbreviations": True,
                                   "linker_name": "umls",
                                   "filter_for_definitions": False,
                                   "no_definition_threshold": 0.85,
                                   "max_entities_per_mention": 1})

        # Load UMLS Semantic Types table
        # see https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/documentation/SemanticTypesAndGroups.html
        self._umls_semantic_types = pd.read_csv(
            "https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt",
            sep="|", names=['abbv', 'tui', 'label'])
        self._non_alphanum_re = re.compile('[\W_]+', re.UNICODE)

    @property
    def model_name(self):
        return self._model

    def extract_entities(self, input_text, input_id="", incl_unlinked_entities=False, output_as_df=False):
        if (not isinstance(input_text, str)) or input_text == "":
            self._log.debug(f"Input text must be a non-empty string: {input_text} ({input_id})")
            return pd.DataFrame() if output_as_df else []
        if input_id == "":
            input_id = shortuuid.ShortUUID().random(length=10)
        input_text = input_text.replace("\n", " ").replace("\t", " ").replace("&nbsp;", " ")
        clean_text = self._non_alphanum_re.sub(" ", input_text)
        clean_text = re.sub(" +", " ", clean_text)
        clean_text = truecase.get_true_case(clean_text)
        entities = []
        doc = self._ner(text=clean_text)
        if len(doc.ents) == 0:
            self._log.debug(f"No named entities found in text: {input_text}")
        for entity in doc.ents:  # Extract named entities and link them to UMLS
            linker = self._ner.get_pipe("scispacy_linker")
            if len(entity._.kb_ents) > 0:
                for umls_entity in entity._.kb_ents:
                    cui, score = umls_entity
                    score = round(score, 3)
                    details = linker.kb.cui_to_entity[umls_entity[0]]
                    self._add_entity_to_output(output=entities, input_id=input_id, input_text=input_text,
                                               entity=entity.text, entity_type=entity.label_, umls_cui=cui,
                                               umls_label=details.canonical_name, umls_semantic_types=details.types,
                                               umls_definition=details.definition,
                                               umls_synonyms=", ".join(details.aliases), umls_mapping_score=score)
            else:
                self._log.debug(f"No UMLS mappings found for entity: {input_text}")
                if incl_unlinked_entities:
                    self._add_entity_to_output(output=entities, input_id=input_id, input_text=input_text,
                                               entity=entity.text, entity_type=entity.label_)
        return self._ner_output(entities, output_as_df)

    def extract_entities_in_list(self, string_list, output_as_df=False, incl_unlinked_entities=False):
        self._log.info(f"Processing list of {len(string_list)} strings...")
        entities = []
        for string in tqdm(string_list):
            entities.extend(self.extract_entities(string, incl_unlinked_entities=incl_unlinked_entities,
                                                  output_as_df=False))
        self._log.info(f"...done")
        return self._ner_output(entities, output_as_df)

    def extract_entities_in_file(self, filepath, input_text_col=None, input_id_col=None, input_col_sep=None,
                                 output_as_df=False, incl_unlinked_entities=False):
        self._log.info(f"Processing file {filepath}...")
        entities = []
        if input_text_col is None:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                entities = self.extract_entities_in_list(lines, incl_unlinked_entities=incl_unlinked_entities,
                                                         output_as_df=False)
        else:
            input_table = pd.read_csv(filepath, sep=input_col_sep)
            for index, row in tqdm(input_table.iterrows(), total=input_table.shape[0]):
                input_text = row[input_text_col]
                input_id = row[input_id_col]
                if not pd.isna(input_text):
                    entities.extend(self.extract_entities(input_text=input_text, input_id=input_id, output_as_df=False,
                                                          incl_unlinked_entities=incl_unlinked_entities))
        self._log.info(f"...done")
        return self._ner_output(entities, output_as_df)

    def _ner_output(self, entities, output_as_df):
        return pd.DataFrame([entity.as_dict() for entity in entities]) if output_as_df else entities

    def _add_entity_to_output(self, output, input_id, input_text, entity, entity_type, umls_cui="", umls_label="",
                              umls_definition="", umls_semantic_types=(), umls_synonyms="", umls_mapping_score=""):
        entity = LinkedNamedEntity(input_id=input_id, input_text=input_text, entity=entity, entity_type=entity_type,
                                   umls_cui=umls_cui, umls_label=umls_label, umls_definition=umls_definition,
                                   umls_synonyms=umls_synonyms, umls_semantic_type_ids=",".join(umls_semantic_types),
                                   umls_semantic_type_labels=self._get_umls_semantic_type_labels(umls_semantic_types),
                                   umls_mapping_score=umls_mapping_score)
        output.append(entity)

    def _get_umls_semantic_type_labels(self, semantic_types):
        semantic_type_labels = ""
        for semantic_type in semantic_types:
            semantic_type_labels_df = self._umls_semantic_types[self._umls_semantic_types["tui"] == semantic_type]
            semantic_type_labels += semantic_type_labels_df["label"].item() + ","
        return semantic_type_labels.rstrip(",")

    @staticmethod
    def ner_models():
        return ["en_ner_bc5cdr_md", "en_ner_jnlpba_md", "en_ner_bionlp13cg_md", "en_ner_craft_md"]

    @staticmethod
    def get_logger(name, level=logging.INFO):
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(name)
        logger.setLevel(level=level)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        logger.propagate = False
        return logger


def do_ner_all_models(input_file, input_text_col, input_id_col, input_col_sep):
    merged_entities_df = pd.DataFrame()
    for ner_model in ScispacyUmlsNer.ner_models():
        scispacy_ner = ScispacyUmlsNer(model=ner_model)
        entities_df = scispacy_ner.extract_entities_in_file(filepath=input_file, output_as_df=True,
                                                            input_text_col=input_text_col, input_id_col=input_id_col,
                                                            input_col_sep=input_col_sep)
        entities_df["ner_model"] = ner_model
        merged_entities_df = pd.concat([merged_entities_df, entities_df], ignore_index=True)
    return merged_entities_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("scispacy_ner")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input file")
    parser.add_argument("-c", "--col", type=str, help="Table column with input text")
    parser.add_argument("-d", "--id", type=str, help="Table column with input text IDs")
    parser.add_argument("-m", "--model", default="all", type=str,
                        help="Name of the scispaCy model to be used")
    args = parser.parse_args()
    input_model = args.model
    input_filepath = args.input

    # prepare output folder and file
    output_dir = os.path.join("..", "..", "output", "scispacy_ner", f"model_{input_model}")
    output_file_path = os.path.join(output_dir, input_filepath.split(os.sep)[-1] + "_entities.tsv")
    os.makedirs(output_dir, exist_ok=True)

    input_file_col_sep = None
    if ".tsv" in input_filepath:
        input_file_col_sep = "\t"
    elif ".csv" in input_filepath:
        input_file_col_sep = ","

    if input_model.lower() == "all":
        detected_entities = do_ner_all_models(input_file=input_filepath, input_id_col=args.id, input_text_col=args.col,
                                              input_col_sep=input_file_col_sep)
    else:
        # instantiate scispacy with the specified model
        my_scispacy = ScispacyUmlsNer(model=input_model)
        detected_entities = my_scispacy.extract_entities_in_file(filepath=input_filepath, output_as_df=True,
                                                                 input_text_col=args.col, input_id_col=args.id,
                                                                 input_col_sep=input_file_col_sep)
    detected_entities.to_csv(output_file_path, sep="\t", index=False)
