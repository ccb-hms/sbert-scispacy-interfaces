"""Provides LinkedNamedEntity class"""


class LinkedNamedEntity:

    def __init__(self, input_id, input_text, entity, entity_type, umls_cui, umls_label, umls_definition, umls_synonyms,
                umls_semantic_type_ids, umls_semantic_type_labels, umls_mapping_score):
        self._input_id = input_id
        self._input_text = input_text
        self._entity = entity
        self._entity_type = entity_type
        self._umls_cui = umls_cui
        self._umls_label = umls_label
        self._umls_definition = umls_definition
        self._umls_synonyms = umls_synonyms
        self._umls_semantic_type_ids = umls_semantic_type_ids
        self._umls_semantic_type_labels = umls_semantic_type_labels
        self._umls_mapping_score = umls_mapping_score

    @property
    def input_id(self):
        return self._input_id

    @property
    def input_text(self):
        return self._input_text

    @property
    def entity(self):
        return self._entity

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def umls_cui(self):
        return self._umls_cui

    @property
    def umls_label(self):
        return self._umls_label

    @property
    def umls_definition(self):
        return self._umls_definition

    @property
    def umls_synonyms(self):
        return self._umls_synonyms

    @property
    def umls_semantic_type_ids(self):
        return self._umls_semantic_type_ids

    @property
    def umls_semantic_type_labels(self):
        return self._umls_semantic_type_labels

    @property
    def umls_mapping_score(self):
        return self._umls_mapping_score

    def as_dict(self):
        return {"InputID": self.input_id,
                "InputText": self.input_text,
                "Entity": self.entity,
                "EntityType": self.entity_type,
                "UMLS.CUI": self.umls_cui,
                "UMLS.Label": self.umls_label,
                "UMLS.Definition": self.umls_definition,
                "UMLS.Synonyms": self.umls_synonyms,
                "UMLS.SemanticTypeIDs": self.umls_semantic_type_ids,
                "UMLS.SemanticTypeLabels": self.umls_semantic_type_labels,
                "UMLS.MappingScore": self.umls_mapping_score}
