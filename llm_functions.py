from accelerate import infer_auto_device_map, init_empty_weights, infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

import torch
import regex as re

import rag_functions as rag


### Initiate LLM ###

def initiate_mistral(path):
    mistral_7b_orca_model_path = path

    global tokenizer, model
    
    config = AutoConfig.from_pretrained(mistral_7b_orca_model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    # print(model._no_split_modules)
    
    device_map = infer_auto_device_map(model, max_memory={0:"2GiB", "cpu":"50GiB"}, 
                                       no_split_module_classes=["MistralDecoderLayer"])
    
    tokenizer = AutoTokenizer.from_pretrained(mistral_7b_orca_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(mistral_7b_orca_model_path, device_map="auto", 
                                                 trust_remote_code=False, revision="gptq-4bit-32g-actorder_True")

def generate_ans(query, context, stream=False, device="cuda"):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None
    
    messages = [
    {"role": "user", "content": f"Act like an intelligent AI assistant. You will answer the questions \
    succinctly based on the context provided. Do not generate information not found in the context. \
    Context: {context}; Question: {query}; Answer: "}
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]

def get_ans(ans):
    ans = re.search(r'(?<=<\|im_end\|>).*(?=<\|im_end\|>)', ans, flags=re.DOTALL).group(0)
    ans2 = re.sub(r'^[\s\n\r]*', '', ans)
    ans3 = re.sub(r'\n', '', ans2)
    return ans3



### Functions to generate press information ###

def generate_details(embedding_function, collection, contexts, subject, 
                     example_1, name_1, ans_1, 
                     example_2, name_2, ans_2, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None

    query = f"Details of {subject}. \
              Subject's nickname, job title, occupation, role? \
              Subject's company, organisation?"

    results = rag.get_results(embedding_function, collection, query, 8)
    input = rag.convert_results(results, contexts)
    
    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. Provide the following \
    information succinctly according to the the preceding case and subject. Do not generate additional \
    information not found in the text. If any of the information is not available, indicate: 'NA'. \
    - State the subject's nickname. \
    - State the subject's occupation. \
    - State the subject's company or organisation. \
    - State the subject's company's nickname."},
    {"role": "user", "content": f"Case: {example_1}; Subject: {name_1}; Answer: "},
    {"role": "assistant", "content": f"{ans_1}"},
    {"role": "user", "content": f"Case: {example_2}; Subject: {name_2}; Answer: "},
    {"role": "assistant", "content": f"{ans_2}"},
    {"role": "user", "content": f"Case: {input}; Subject: {subject}; Answer: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]

def get_details(info, subject):
    ans = re.search(f'Subject: {subject}.*', info, flags=re.DOTALL).group(0)
    ans = re.search(r'(?<=<\|im_end\|>).*(?=<\|im_end\|>)', ans, flags=re.DOTALL).group(0)
    ans = re.sub(r'^[\s\n\r]*', '', ans)
    ans = re.sub(r'\n', '', ans)
    details_lst = [i.split(':') for i in ans.split('- ')[1:]]
    details_dic = {i: j for i, j in details_lst}
    return details_dic

def generate_crime(embedding_function, collection, contexts, subject, 
                     example_1, name_1, ans_1, 
                     example_2, name_2, ans_2, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None

    query = f"What is {subject} charged with? How many charges? \
              Is {subject} the giver/receiver in the corruption? \
              What is the form of gratification? \
              Is {subject} guilty?"

    results = rag.get_results(embedding_function, collection, query, 8)
    input = rag.convert_results(results, contexts)
    
    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. Provide the following \
    information succinctly according to the the preceding case and subject. Do not generate additional \
    information not found in the text. If any of the information is not available, indicate: 'NA'. \
    - State if the subject is a giver or receiver of the bribe. \
    - State the number of charges the subject was faced with. \
    - State the form of gratification. \
    - State the charge given to the subject. \
    - State if the subject admitted guilty."},
    {"role": "user", "content": f"Case: {example_1}; Subject: {name_1}; Answer: "},
    {"role": "assistant", "content": f"{ans_1}"},
    {"role": "user", "content": f"Case: {example_2}; Subject: {name_2}; Answer: "},
    {"role": "assistant", "content": f"{ans_2}"},
    {"role": "user", "content": f"Case: {input}; Subject: {subject}; Answer: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]

def get_crime(info, subject):
    ans = re.search(f'Subject: {subject}.*', info, flags=re.DOTALL).group(0)
    ans = re.search(r'(?<=<\|im_end\|>).*(?=<\|im_end\|>)', ans, flags=re.DOTALL).group(0)
    ans = re.sub(r'^[\s\n\r]*', '', ans)
    ans = re.sub(r'\n', '', ans)
    details_lst = [i.split(':') for i in ans.split('- ')[1:]]
    details_dic = {i: j for i, j in details_lst}
    return details_dic

def generate_testimony(embedding_function, collection, contexts, subject, 
                       example_1, name_1, example_testimony_1, 
                       example_2, name_2, example_testimony_2, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None

    query = f"According to {subject}, what was the amount of bribe that was paid or received?"

    results = rag.get_results(embedding_function, collection, query, 5)
    input = rag.convert_results(results, contexts)
    
    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. Keep your answer within two succinct sentences."},
    # {"role": "user", "content": f"Case: {example_1}; Subject: {name_1}; Answer: "},
    # {"role": "assistant", "content": f"{example_testimony_1}"},
    # {"role": "user", "content": f"Case: {example_2}; Subject: {name_2}; Answer: "},
    # {"role": "assistant", "content": f"{example_testimony_2}"},
    {"role": "user", "content": f"According to the testification by {subject}, what was the amount of bribe that was paid \
    or received? Keep your answer within two succinct sentences. If the bribe amount is not available, indicate: 'NA'. \
    Case: {input}; Subject: {subject}; Answer: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)
    
    return decoded[0]

def get_testimony(info, subject):
    ans = re.search(f'Subject: {subject}.*', info, flags=re.DOTALL).group(0)
    ans = re.search(r'(?<=<\|im_end\|>).*(?=<\|im_end\|>)', ans, flags=re.DOTALL).group(0)
    ans = re.sub(r'^[\s\n\r]*', '', ans)
    ans = re.sub(r'\n', '', ans)
    return ans

def generate_passport(embedding_function, collection, contexts, subject, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None

    query = f"Details of {subject}: age, years old, gender, nationality" 

    results = rag.get_results(embedding_function, collection, query, 5)
    input = rag.convert_results(results, contexts)
    
    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. You will answer my question \
    succintly and accurately using the preceding context. Do not generate your own additional \
    information. \
    What is the age, gender and nationality of the subject? \
    Answer in the following format: <age-year-old> <gender> <nationality> \
    If any of the information is not available, replace it with: <NA>"},
    {"role": "user", "content": f"Context: f'{input}; Subject: {subject}; Answer: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]

def get_passport(info, subject):
    ans = re.search(f'Subject: {subject}.*', info, flags=re.DOTALL).group(0)
    ans2 = re.search(r'(?<=<\|im_end\|>).*(?=<\|im_end\|>)', ans, flags=re.DOTALL).group(0)
    ans3 = re.sub('^\s+', '', ans2)
    return ans3

def generate_summary(embedding_function, collection, contexts, details, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None
    
    query = f"Who are the receivers and givers of this corruption case? \
              What crimes did {', '.join(details.keys())} commit? \
              How much was the bribe or gratification amount?"
    results = rag.get_results(embedding_function, collection, query, 5)
    input = rag.convert_results(results, contexts)

    ## Need specify names and roles of the individuals coz RAG's context might not have these info.
    names = list(details.keys())
    lastnames = [f"('{details[i]['Nick']}')" for i in details.keys()]
    fullnames = [' '.join(i) for i in (zip(names, lastnames))]
    roles = [details[i]['Role'] for i in details.keys()]
    combined = [': '.join(i) for i in zip(fullnames, roles)]
    
    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. Based on the provided case, \
    you will succinctly summarise the corruption incident. Include all the details about the \
    dates of the crime, the type of gratification, the bribe amount and any other criminal acts. \
    These are the individuals and their roles. {'. '.join(combined)}."},
    {"role": "user", "content": f"Case: f'{input}; Summary: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]

def get_summary(info):
    ans = re.search(r'(?<=<\|im_end\|>[\s\n]+).*(?=<\|im_end\|>)', info, flags=re.DOTALL).group(0)
    ans2 = re.sub('^\s+', '', ans)
    return ans2

def generate_chargeDate(embedding_function, collection, contexts, subject, details, stream=False):
    torch.cuda.empty_cache()
    streamer = TextStreamer(tokenizer)
    if not stream:
        streamer = None
    
    query = f"When was {subject} ({details[subject]['Nick']}) charged in court? \
              When did {subject} ({details[subject]['Nick']}) receive verdict/sentencing from the judge?"
    results = rag.get_results(embedding_function, collection, query, 5)
    input = rag.convert_results(results, contexts)

    messages = [
    {"role": "system", "content": f"Act like an intelligent AI assistant. You will answer my question \
    succintly and accurately using the preceding case. Do not generate your own additional \
    information. \
    Answer me in 1 word, in the following format: <the date {subject} ({details[subject]['Nick']}) was charged/convicted> \
    If the information is not available, return: NA"},
    {"role": "user", "content": f"Case: f'{input}; Answer: "},
    ]
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to("cuda")
    
    generated_ids = model.generate(
        model_inputs, 
        temperature=0.2, 
        do_sample=True, 
        top_p=0.2, 
        top_k=40, 
        max_new_tokens=1024,
        streamer=streamer
    )
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]

def get_chargeDate(info):
    ans = re.search(r'(?<=<\|im_end\|>[\s\n]+).*(?=<\|im_end\|>)', info, flags=re.DOTALL).group(0)
    ans2 = re.sub('^\s+', '', ans)
    return ans2


