## Press release generation
<br> <br> 
press_generation.ipynb
> the main notebook that generates a formal press according to the input case law
> the whole processes harnesses document ingestion, retrieval augmented generation, document analysis and large language modelling techniques to generate the report
<br> <br>

rag_functions.py
> contains methods under the rag class which allows for ingestion and segmentation of the various sections of a documents
> the segmented paragraphs are then embedded via chromadb, which a language model of your choice
<br> <br>

llm_functions.py
> contains various functions to pre-generate paragraphs of critical information with regards to the case law
> the llm model used is mistral-7b unquantised
> prompt engineering is embedded within each function to engineer the output in a specific format
<br> <br>
prompt_examples.json
> a dictionary of examples to be used within the prompt engineering
> should be amended according to the desired report output format
