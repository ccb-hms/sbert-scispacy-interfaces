from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scispacy_ner import ScispacyUmlsNer

__version__ = "0.1.0"


def get_pdf_chunks(pdf_file):
    loader = PyPDFLoader(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    chunks = loader.load_and_split(text_splitter)
    output_data = []
    for chunk in chunks:
        chunk_text = chunk.page_content
        chunk_text = chunk_text.replace("-\n", "")
        chunk_text = chunk_text.replace("\n", " ")
        output_data.append(chunk_text)
    return output_data


if __name__ == '__main__':
    data = "data/"
    pdf1 = "AMIE.pdf"
    pdf2 = "1-s2.0-S1538544221000821-main.pdf"

    my_pdf_file_path = data + pdf2
    my_scispacy = ScispacyUmlsNer()
    my_pdf_chunks = get_pdf_chunks(my_pdf_file_path)

    entities_df = my_scispacy.extract_entities_in_list(my_pdf_chunks, output_as_df=True)
    entities_df.to_csv(f"output/{pdf2}.tsv", sep="\t", index=False)
