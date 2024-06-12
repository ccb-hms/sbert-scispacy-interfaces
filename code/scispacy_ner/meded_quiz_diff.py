import os
import logging
import argparse
import pandas as pd
from scispacy_ner import ScispacyUmlsNer

__version__ = "0.1.0"


class GradedAnswer:

    def __init__(self, quiz_id, submission_id, question_name, entity, entity_type, entity_cui, grade):
        self._quiz_id = quiz_id
        self._submission_id = submission_id
        self._question_name = question_name
        self._entity = entity
        self._entity_type = entity_type
        self._entity_cui = entity_cui
        self._grade = grade

    @property
    def quiz_id(self):
        return self._quiz_id

    @property
    def submission_id(self):
        return self._submission_id

    @property
    def question_name(self):
        return self._question_name

    @property
    def entity(self):
        return self._entity

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def entity_cui(self):
        return self._entity_cui

    @property
    def grade(self):
        return self._grade

    def as_dict(self):
        return {
            'quiz_id': self._quiz_id,
            'submission_id': self._submission_id,
            'question_name': self._question_name,
            'entity': self._entity,
            'entity_type': self._entity_type,
            'entity_cui': self._entity_cui,
            'grade': self._grade
        }


LOG = ScispacyUmlsNer.get_logger("meded.quiz.diff", logging.INFO)


def compare_quiz_answers(model_answers, student_answers, output_as_df=False):
    graded_answers = []
    quiz_ids = model_answers["quiz_id"].unique()
    for quiz_id in quiz_ids:
        quiz_model_answers = model_answers[model_answers["quiz_id"] == quiz_id]
        quiz_student_answers = student_answers[student_answers["quiz_id"] == quiz_id]
        quiz_submissions = quiz_student_answers["submission_id"].unique()
        LOG.debug(f"Quiz {quiz_id}")
        LOG.debug(f" {quiz_model_answers.shape[0]} model answers in this quiz")
        LOG.debug(f" {quiz_submissions.shape[0]} student answers in this quiz")
        for submission_id in quiz_submissions:
            LOG.debug(f" Submission {submission_id}")
            quiz_student_submission = quiz_student_answers[quiz_student_answers["submission_id"] == submission_id]
            for question in quiz_model_answers["question_name"].unique():
                LOG.debug(f"  {question}")
                model_answer = quiz_model_answers[quiz_model_answers["question_name"] == question]
                model_answer_cuis = model_answer["UMLS.CUI"].unique()
                LOG.debug(f"    Model answer CUIS: {model_answer_cuis}")
                student_answer = quiz_student_submission[quiz_student_submission["question_name"] == question]
                student_answer_cuis = student_answer["UMLS.CUI"].unique()
                LOG.debug(f"    Student answer CUIS: {student_answer_cuis}")
                # get entities in the model answer that are not in the student answer
                graded_answers.extend(diff_answer(model_answer_cuis, student_answer_cuis, model_answer, "-",
                                                  quiz_id, submission_id, question))
                # get entities in the student answer that are not in the model answer
                graded_answers.extend(diff_answer(student_answer_cuis, model_answer_cuis, student_answer, "+",
                                                  quiz_id, submission_id, question))
    return pd.DataFrame([answer.as_dict() for answer in graded_answers]) if output_as_df else graded_answers


def diff_answer(left_cuis, right_cuis, answer_df, grade, quiz_id, submission_id, question):
    graded_answers = []
    for left_cui in left_cuis:
        if left_cui not in right_cuis:
            LOG.debug(f"     {grade} entity {left_cui}")
            cui_row = answer_df[answer_df["UMLS.CUI"] == left_cui]
            entity = cui_row["Entity"].unique()[0]
            entity_type = cui_row["EntityType"].unique()[0]
            graded_answer = GradedAnswer(quiz_id=quiz_id, submission_id=submission_id, question_name=question,
                                         entity=entity, entity_type=entity_type, entity_cui=left_cui, grade=grade)
            graded_answers.append(graded_answer)
    return graded_answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser("meded_quiz_diff")
    parser.add_argument("-d", "--dir", required=True, type=str,
                        help="Directory containing the tables of entities detected in the model and student answers")
    args = parser.parse_args()
    ner_data_folder = args.dir
    for model in ScispacyUmlsNer.ner_models():
        model_answers_df = pd.read_csv(f"{ner_data_folder}{os.sep}ner_{model}_model_answers.tsv", sep="\t")
        student_answers_df = pd.read_csv(f"{ner_data_folder}{os.sep}ner_{model}_student_answers.tsv", sep="\t")
        diff = compare_quiz_answers(model_answers=model_answers_df, student_answers=student_answers_df,
                                    output_as_df=True)
        diff.to_csv(f"{ner_data_folder}{os.sep}ner_{model}_quiz_diff.tsv", sep="\t", index=False)
