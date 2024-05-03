from langchain.text_splitter import MarkdownTextSplitter
import pdf4llm
import fitz
import re

__version__ = "0.1.0"


#   File "/usr/local/lib/python3.9/site-packages/fitz/__init__.py", line 1, in <module>
#     from frontend import *
# ModuleNotFoundError: No module named 'frontend'
#
# --> pip3.9 install frontend
#
# Traceback (most recent call last):
#   File "/Users/rsgoncalves/Documents/Harvard/Workspace/sbert-scispacy-interfaces/file_handler.py", line 4, in <module>
#     import pdf4llm
#   File "/usr/local/lib/python3.9/site-packages/pdf4llm/__init__.py", line 1, in <module>
#     from .helpers.pymupdf_rag import to_markdown
#   File "/usr/local/lib/python3.9/site-packages/pdf4llm/helpers/pymupdf_rag.py", line 41, in <module>
#     import fitz
#   File "/usr/local/lib/python3.9/site-packages/fitz/__init__.py", line 1, in <module>
#     from frontend import *
#   File "/usr/local/lib/python3.9/site-packages/frontend/__init__.py", line 1, in <module>
#     from .events import *
#   File "/usr/local/lib/python3.9/site-packages/frontend/events/__init__.py", line 1, in <module>
#     from .clipboard import *
#   File "/usr/local/lib/python3.9/site-packages/frontend/events/clipboard.py", line 2, in <module>
#     from ..dom import Event
#   File "/usr/local/lib/python3.9/site-packages/frontend/dom.py", line 439, in <module>
#     from . import dispatcher
#   File "/usr/local/lib/python3.9/site-packages/frontend/dispatcher.py", line 15, in <module>
#     from . import config, server
#   File "/usr/local/lib/python3.9/site-packages/frontend/server.py", line 24, in <module>
#     app.mount(config.STATIC_ROUTE, StaticFiles(directory=config.STATIC_DIRECTORY), name=config.STATIC_NAME)
#   File "/usr/local/lib/python3.9/site-packages/starlette/staticfiles.py", line 57, in __init__
#     raise RuntimeError(f"Directory '{directory}' does not exist")
# RuntimeError: Directory 'static/' does not exist
#
# --> mkdir static
#
# ModuleNotFoundError: No module named 'tools'
#
# --> pip install tools
#
# Traceback (most recent call last):
#   File "/Users/rsgoncalves/Documents/Harvard/Workspace/sbert-scispacy-interfaces/file_handler_rog.py", line 2, in <module>
#     import pdf4llm
#   File "/usr/local/lib/python3.9/site-packages/pdf4llm/__init__.py", line 1, in <module>
#     from .helpers.pymupdf_rag import to_markdown
#   File "/usr/local/lib/python3.9/site-packages/pdf4llm/helpers/pymupdf_rag.py", line 43, in <module>
#     if fitz.pymupdf_version_tuple < (1, 24, 0):
# AttributeError: module 'fitz' has no attribute 'pymupdf_version_tuple'


def get_pdf_chunks(pdf_file):
    md_text = pdf4llm.to_markdown(pdf_file)
    splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=0)
    return splitter.create_documents([md_text])


def get_pdf_images(pdf_file):
    doc = fitz.open(pdf_file)  # open a document
    images = []
    for page_index in range(len(doc)):  # iterate over pdf pages
        page = doc[page_index]
        image_list = page.get_images()
        if image_list:
            images.extend(image_list)
            print(f"Found {len(image_list)} images on page {page_index}")
        else:
            print("No images found on page", page_index)
    return images


def get_pdf_tables(pdf_file):
    doc = fitz.open(pdf_file)  # open document
    tables = []
    for page_index in range(len(doc)):  # iterate over pdf pages
        page = doc[page_index]
    page = doc[9]  # get the 1st page of the document
    tabs = page.find_tables()  # locate and extract any tables on page
    print(f"{len(tabs.tables)} found on {page}")  # display number of found tables
    if tabs.tables:  # at least one table found?
        print(tabs[0].extract())  # print content of first table/

    links = page.get_links()
    paras = page.get_text("blocks")


if __name__ == '__main__':
    amie = "data/AMIE.pdf"
    chunks = get_pdf_chunks(amie)
    print(chunks)
