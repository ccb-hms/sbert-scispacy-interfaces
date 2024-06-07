from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llmsherpa.readers import LayoutPDFReader

__version__ = "0.1.0"


# CCB Ollama endpoint
ccb_endpoint = 'http://compute-gc-17-255.o2.rc.hms.harvard.edu:11434'

# create embedding model
embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# create LLM
llm = Ollama(model="llama2", base_url=ccb_endpoint, temperature=0)
Settings.llm = llm
Settings.embed_model = embed_model
# Settings.chunk_size = 512


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

def get_pdf_chunks_llmsherpa(pdf_url="https://arxiv.org/pdf/2401.05654"):
    print(f"loading {pdf_url}")
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(pdf_url)

    index = VectorStoreIndex([])
    counter = 1
    for chunk in doc.chunks():
        print(f"Chunk {counter}: {chunk.to_context_text}")
        index.insert(Document(text=chunk.to_context_text(), extra_info={}))
        counter += 1
    query_engine = index.as_query_engine()

    # Let's run one query
    response = query_engine.query("What is AMIE and what is it used for?")
    print(response)


if __name__ == '__main__':
    data_folder = "data/"
    pdf1 = "AMIE.pdf"
    pdf2 = "1-s2.0-S1538544221000821-main.pdf"
    pdf = pdf1

    my_pdf_file_path = data_folder + pdf
    my_pdf_chunks = get_pdf_chunks_llmsherpa(my_pdf_file_path)
    # my_scispacy = ScispacyUmlsNer()
    # entities_df = my_scispacy.extract_entities_in_list(my_pdf_chunks, output_as_df=True, incl_ungrounded_entities=False)
    # entities_df.to_csv(f"output/{pdf}.tsv", sep="\t", index=False)
