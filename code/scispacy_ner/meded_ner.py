import os
import argparse
import pandas as pd
from tqdm import tqdm
from scispacy_ner import ScispacyUmlsNer

__version__ = "0.1.1"


def load_json_file_as_df(json_file_path):
    df_from_json = pd.read_json(json_file_path)
    df_from_json = df_from_json.loc[(df_from_json['question_type'] == 'essay_question') &
                                    (df_from_json['quiz_title'].str.contains("Consolidation"))]
    return df_from_json


def do_ner(course_table, ner):
    entities_in_model_answers = pd.DataFrame()
    entities_in_student_answers = pd.DataFrame()
    quizzes = course_table['quiz_id'].unique()
    for quiz in tqdm(quizzes):
        quiz_questions = {}
        quiz_answers = course_table[course_table['quiz_id'] == quiz]
        for index, row in quiz_answers.iterrows():
            history_id = row["history_id"]
            submission_id = row["submission_id"]
            question_name = row["question_name"]
            question_answer = row["question_answer"]
            student_answer = row["student_answer"]

            # do NER over the model answer if it has not been processed already
            if question_name not in quiz_questions:
                question_id = f"quiz-{quiz}_model_answer_{question_name}"
                question_entities = _do_ner(ner, input_text=question_answer, input_id=question_id,
                                            quiz_id=quiz, question_name=question_name)
                entities_in_model_answers = pd.concat([entities_in_model_answers, question_entities], ignore_index=True)

            # do NER over the student answer
            answer_id = f"quiz-{quiz}_submission-{submission_id}_question-{question_name}"
            student_entities = _do_ner(ner, input_text=student_answer, input_id=answer_id, quiz_id=quiz,
                                       question_name=question_name, submission_id=submission_id, history_id=history_id)
            entities_in_student_answers = pd.concat([entities_in_student_answers, student_entities], ignore_index=True)
    return entities_in_student_answers, entities_in_model_answers


def _do_ner(ner, input_text, input_id, quiz_id, question_name, submission_id="", history_id=""):
    entities_df = ner.extract_entities(input_text=input_text, input_id=input_id, output_as_df=True)
    entities_df = entities_df.drop_duplicates()
    _add_details_to_df(entities_df, quiz_id=quiz_id, question_name=question_name,
                       submission_id=submission_id, history_id=history_id)
    return entities_df


def _add_details_to_df(df, quiz_id, question_name, submission_id="", history_id=""):
    df["quiz_id"] = quiz_id
    df["question_name"] = question_name
    if submission_id != "":
        df["submission_id"] = submission_id
    if history_id != "":
        df["history_id"] = history_id
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser("meded_ner")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input JSON file")
    parser.add_argument("-m", "--model", type=str, help="Name of the scispaCy model to be used")
    input_args = parser.parse_args()
    input_json_file = input_args.input

    # list NER models of interest -- probably should run each model individually via a shell script
    ner_models = ["en_ner_jnlpba_md", "en_ner_bc5cdr_md", "en_ner_bionlp13cg_md", "en_ner_craft_md"]
    # ner_model = input_args.model

    # specify output folder based on the course name
    course_name = os.path.basename(os.path.dirname(input_json_file))
    output_dir = os.path.join("output", course_name)
    os.makedirs(output_dir, exist_ok=True)

    # load the input JSON file into a pandas dataframe and serialize it
    course_data = load_json_file_as_df(input_json_file)
    course_data.to_csv(f"{output_dir}{os.sep}essay_data.tsv", sep="\t", index=False)

    for ner_model in ner_models:
        # instantiate our scispacy named entity recognizer with the given model name
        scispacy = ScispacyUmlsNer(model=ner_model)

        # do NER over the loaded course data
        student_entities_df, model_entities_df = do_ner(course_data, scispacy)

        # save the obtained data frames of entities detected in the student answers and of entities in the model answers
        student_entities_df.to_csv(f"{output_dir}{os.sep}ner_{ner_model}_student_answers.tsv", sep="\t", index=False)
        model_entities_df.to_csv(f"{output_dir}{os.sep}ner_{ner_model}_model_answers.tsv", sep="\t", index=False)
