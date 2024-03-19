from unstructured.partition.auto import partition_pdf
from unstructured.partition.auto import partition

import PyPDF2
from PyPDF2 import PdfWriter, PdfReader
from PIL import Image
from pdf2image import convert_from_path

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import numpy as np
from sentence_transformers import SentenceTransformer

import pickle
import fitz
import regex as re
import os


### Initiate ###

def initiate():
    current_directory = os.getcwd()
    folders = [r'figures', r'pages']
    for f in folders:
        if not os.path.exists(os.path.join(current_directory, f)):
            os.makedirs(f)
    
    fig_path = os.path.join(os.getcwd(), 'figures')
    pages_path = os.path.join(os.getcwd(), 'pages')
    
    for f in [fig_path, pages_path]:
        if not os.path.exists(f):
            os.makedirs(f)

    if not os.path.exists("pickles"):
        os.makedirs("pickles")


### Unstructured ###

def get_raw_elements(files, generate_new=False):
    if not generate_new:
        try:
            with open("pickles/raw_elements.pickle", "rb") as f:
                return pickle.load(f)
        except:
            return "No raw_elements.pickle detected. Try generate_new=True."
    
    raw_elements = {}
    for idx in files.keys():
        raw_elements[idx] = partition_pdf(filename=files[idx], strategy='hi_res',
                                          infer_table_structure=True, extract_images_in_pdf=True)
    with open("pickles/raw_elements.pickle", "wb") as f:
        pickle.dump(raw_elements, f)
    return raw_elements


### Functions to identify section headers ###

## Detect font variations
def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)


## for searchable PDFs (not scannable)
def retrieveSections(doc):
    sections = {}

    for page in range(len(doc)):
        blocks = doc[page].get_text("dict", flags=4)["blocks"]
        ## Iterate through text blocks and ignore image blocks (b["type"] == 1)
        for b in blocks:
            if b["type"] == 1:
                continue

            ## Iterate through text lines
            for l in b["lines"]:
                currentLineBold = True
                currentLine = ''

                ## Iterate through text spans (spans are characters with same font properties)
                for s in l["spans"]:

                    ## Ignore blank spaces
                    if s['text'].isspace():
                        continue
                    
                    ## Remove trailing white spaces in chapter
                    text = re.sub(r'\s+$', ' ', s['text'])
                    currentLine += text

                    ## Analyse font properties of current span
                    font_properties = {'font': s['font'], 'flags': flags_decomposer(s['flags']), 
                                       'size': s['size'], 'color': s['color']}

                    ## Skip current line as long as 1 span is not bold
                    if "bold" not in font_properties['flags']:
                        currentLineBold = False
                        continue
                    continue
                
                ## Retrieve only lines with all spans bold
                if currentLineBold and currentLine:
                    currentLine = re.sub(r'\s+$', '', currentLine)
                    sections.update({currentLine: page})
    
    return sections


### Functions to process PDFs ###

def clean_elements(elements):
    new_elements = []
    all_text = [e.to_dict()['text'] for e in elements]
    repeated = [i for i in set(all_text) if all_text.count(i) > 5]

    for e in elements:
        info = e.to_dict()
        ## Ignore content page, page numbers
        if info["type"] == "UncategorizedText":
            # except number bullet points e.g. "3. xxx, 4. xxx"
            if not re.findall("[0-9]+[.]", info["text"]):
                continue
        
        ## Ignore header image
        # if 'VA TECH HYDRO' in info['text']:
        #     continue
        ## Ignore headers/footers (images)

        ## Ignore other repeated texts
        if info["text"] in repeated:
            continue

        new_elements.append(e)
    
    return new_elements

def analyse(elements, sections):

    elementsType = [e.to_dict()["type"] for e in elements]

    contexts = []
    meta = []             ## list of page no. and sections corresponding to each item in contexts 
    figStorage = {}       ## keys: section; values: list of dictionaries containing image/table info (idx, caption)
    sectionPages = {}     ## keys: section; values: dictionary of firstPage and lastPage numbers
    currentSection = None
    listItems = ''        ## temp variable

    for i in range(len(elements)):
        info = elements[i].to_dict()
        currentPage = info['metadata']['page_number'] - 1  ## unstructured page number starts from 1

        ## If current page contains section header, scan through every text to find section header
        if currentPage in sections.values():
            if info["text"] in sections:
                currentSection = info["text"]
                sectionPages[currentSection] = {"firstPage": currentPage, "lastPage": currentPage}
                continue 
        
        ## Skip first few elements when a section header is not yet identified
        if not currentSection:
            continue
        
        ## Combining bullet points into single context
        if elementsType[i] == "ListItem":
            # If next line is a bullet point as well, append them tgt b4 adding to contexts
            if (i < len(elements)) and (elementsType[i+1] == "ListItem"):
                listItems += info["text"] + ' '
                continue
            # If next line no longer bullet point, check if listItems has content
            if listItems:
                listItems += info["text"]
                contexts.append(listItems)
                listItems = ''
            # If this is an isolated single bullet point
            else:
                contexts.append(info["text"])
            meta.append({"page": currentPage, "section": currentSection})
            sectionPages[currentSection]["lastPage"] = currentPage
            continue
        
        ## Add captions labelled NarrativeText to image/table_storage, assume it comes after image and is empty
        if elementsType[i] == "NarrativeText":
            # if i != (0 or len(elements)) and ("Image" or "Table") in [elementsType[i-1], elementsType[i+1]]:
            if (i != 0)  and (elementsType[i-1] in ["Image", "Table"]):
                if ("Figure" in info["text"]) or ("Table" in info["text"]):
                    if not figStorage[currentSection][-1]["caption"]:  ## caption currently empty
                        figStorage[currentSection][-1].update({"caption": info["text"]})
                continue
        
        ## Append narrative text
        if elementsType[i] == "NarrativeText":
            # skip if only 3 words in text (future implementation: ignore only if 3 items are not words)
            if len(info["text"].split(" ")) > 3:
                contexts.append(info["text"])
                meta.append({"page": currentPage, "section": currentSection})
                sectionPages[currentSection]["lastPage"] = currentPage
            continue
        
        ## Save figure, Add idx to figStorage (assumed figure comes before caption)
        if elementsType[i] in ("Image", "Table"):
            if currentSection in figStorage:
                figStorage[currentSection].append({"idx": i, "caption": None})
            else:
                figStorage[currentSection] = [{"idx": i, "caption": None}]

        ## Add caption to figStorage, assume it comes after image
        if elementsType[i] == "FigureCaption":
            # Add caption as narrative text if no Figure before/after caption
            if (i not in (0, len(elements))) and (elementsType[i-1] not in ("Image", "Table")) and (elementsType[i+1] not in ("Image", "Table")):
                contexts.append(info["text"])
                meta.append({"page": currentPage, "section": currentSection})
                sectionPages[currentSection]["lastPage"] = currentPage
            if elementsType[i-1] in ("Image", "Table"):
                figStorage[currentSection][-1].update({"caption": info["text"]})
    
    return {"contexts": contexts, "meta": meta,
            "figStorage": figStorage, "currentSection": currentSection, 
            "sectionPages": sectionPages}


### Generate information of PDFs ###

def get_info(files, raw_elements, generate_new=False):
    if not generate_new:
        try:
            with open("pickles/info.pickle", "rb") as f:
                return pickle.load(f)
        except:
            return "No info.pickle detected. Try generate_new=True."
    
    info = dict()
    for idx in files.keys():
        doc = fitz.open(files[idx])
        sections = retrieveSections(doc)
        elements = clean_elements(raw_elements[idx])
        anal = analyse(elements, sections)
        info[idx] = dict(zip(["contexts", "meta", "figStorage", "sectionPages"], 
                            (anal["contexts"], anal["meta"], anal["figStorage"], anal["sectionPages"])))

    ## Saving contexts into pickle
    with open("pickles/info.pickle", "wb") as f:
        pickle.dump(info, f)

    return info


### Functions to save and print figures ###

def save_fig(elements, idx, padding=30):
    ## get coordinates
    upper_left = elements[idx].to_dict()['metadata']['coordinates']['points'][1]
    lower_right = elements[idx].to_dict()['metadata']['coordinates']['points'][3]

    ## get scale
    reader = PdfReader(file)
    page_number = elements[idx].metadata.page_number - 1
    page = reader.pages[page_number]
    pypdf2_width = page.cropbox.lower_right[0]
    pypdf2_height = float(page.cropbox.upper_right[1])
    unstructured_width = elements[idx].to_dict()['metadata']['coordinates']['layout_width']
    scale = float(unstructured_width / pypdf2_width)

    ## extract image (via saving locally)
    writer = PdfWriter()
    # padding = 30
    page.mediabox.lower_left = ((upper_left[0] / scale) - padding, (pypdf2_height - upper_left[1] / scale) - padding)
    page.mediabox.upper_right = ((lower_right[0] / scale) + padding, (pypdf2_height - lower_right[1] / scale) + padding)
    writer.add_page(page)

    with open(os.path.join(fig_path, f'fig_{idx}.pdf'), 'wb') as fig_info:
        writer.write(fig_info)

def print_fig(elements, idx):
    save_fig(elements, idx)
    fig = convert_from_path(os.path.join(fig_path, f'fig_{idx}.pdf'), 500)
    return fig[0]

def save_pages(pageNum: list):
    writer = PdfWriter()
    reader = PdfReader(file)
    with open(os.path.join(pages_path, f'pages.pdf'), 'wb') as f:
        for p in pageNum:
            page = reader.pages[p]
            writer.add_page(page)
        writer.write(f)

def print_pages(pageNum: list):
    save_pages(pageNum)
    page = convert_from_path(os.path.join(pages_path, f'pages.pdf'), 500)
    return page


### Chromadb ###

class MyEmbeddingFunction(EmbeddingFunction[Documents]):
    ## Update to chromadb requires specifying input type: Documents/Images
    ## Unless using chroma's default embedding functions
    
    def __init__(self, embedding_model_path, query=False):
        self.query = query
        self.embedding_model_path = embedding_model_path
        
    def __call__(self, input: Documents) -> Embeddings:
        maxLength = max([len(para) for para in input])

        # 4 characters = 1 token, check this part (split into smaller para if need?)
        embedding_model = SentenceTransformer(self.embedding_model_path)
        embedding_model.max_seq_length = min(maxLength//4 + 20, 512)
    
        embeddings = embedding_model.encode(input)

        ## Take note that gte-large model takes maximum 512 input tokens, else truncated
        if not self.query:
            print(f'Longest context = {maxLength} char = {round(maxLength/4)} tokens')
            print(f'Shape of embeddings: {embeddings.shape}\n')
        
        return embeddings

def generateCollection(embedding_model, name, contexts, metadata, new=False,):
    chroma_client = chromadb.PersistentClient(path=os.path.join('chroma'))

    try:
        if new:
            chroma_client.delete_collection(name=name)
            print(f'Stored collection deleted: "{name}"')
        else:
            collection = chroma_client.get_collection(name=name, embedding_function=embedding_model)
            print(f'Stored collection retrieved: "{name}"')
            return collection
    except:
        pass
    
    collection = chroma_client.create_collection(name=name, embedding_function=embedding_model, 
                                                 metadata={"hnsw:space": "cosine"})
    print(f'New collection created: "{name}"')
    
    embeddings = embedding_model(contexts)
    collection.add(
        embeddings=embeddings.tolist(), 
        metadatas=metadata, 
        ids=[f'{i}' for i in range(len(contexts))])
    
    return collection

## Function to get/generate collections of all documents
def get_collections(embedding_function, info, generate_new=False):
    collections = dict()
    for i in info.keys():
        collections[i] = generateCollection(embedding_function, f'data_{i}', info[i]["contexts"], 
                                            info[i]["meta"], new=generate_new)
    return collections


### Query chroma ###

def get_results(embedding_function, collection, query, numResults):
    query_embeddings = embedding_function(query.lower())
    results = collection.query(
                query_embeddings=query_embeddings.tolist(), 
                n_results=numResults)
    return results

def convert_results(results, contexts):
    input = ''
    for para in results['ids'][0]:
        input += ' \n' + contexts[int(para)]
    return input














