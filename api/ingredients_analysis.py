import logging
import traceback
import sys
from functools import wraps
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from calc_cosine_similarity import find_relevant_file_paths

app = FastAPI(debug = True)

def create_assistant_and_embeddings(client, embeddings_file_list):
    assistant1 = client.beta.assistants.create(
      name="Processing Level",
      instructions="You are an expert dietician. Use your knowledge base to answer questions about the processing level of food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )

      # Create a vector store
    vector_store1 = client.beta.vector_stores.create(name="Processing Level Vec")
    
    # Ready the files for upload to OpenAI
    file_paths = ["Processing_Level.docx"]
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch1 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store1.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch1.status)
    print(file_batch1.file_counts)

    #Processing Level
    assistant1 = client.beta.assistants.update(
      assistant_id=assistant1.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store1.id]}},
    )

    embeddings_titles_list = []
    for embeddings_file in embeddings_file_list:
      embeddings_titles = []
  
      print(f"Reading {embeddings_file}")
      # Load both sentences and embeddings
      with open(embeddings_file, 'rb') as f:
          loaded_data = pickle.load(f)
          embeddings_titles = loaded_data['embeddings']
          embeddings_titles_list.append(embeddings_titles)

    return assistant1, embeddings_titles_list

def get_files_with_ingredient_info(ingredient, embeddings_titles_list, N=1):

    embeddings_titles_1 = embeddings_titles_list[0]
    with open('titles.txt', 'r') as file:
        lines = file.readlines()
    
    titles = [line.strip() for line in lines]
    folder_name_1 = "articles"
    #Apply cosine similarity between embedding of ingredient name and title of all files
    file_paths_abs_1, file_titles_1, refs_1 = find_relevant_file_paths(ingredient, embeddings_titles_1, titles, folder_name_1, journal_str = ".ncbi.", N=N)

    embeddings_titles_2 = embeddings_titles_list[1]
    with open('titles_harvard.txt', 'r') as file:
        lines = file.readlines()
    
    titles = [line.strip() for line in lines]
    folder_name_2 = "articles_harvard"
    #Apply cosine similarity between embedding of ingredient name and title of all files
    file_paths_abs_2, file_titles_2, refs_2 = find_relevant_file_paths(ingredient, embeddings_titles_2, titles, folder_name_1, N=N)

    #Fine top N titles that are the most similar to the ingredient's name
    #Find file names for those titles
    file_paths = []
    refs = []
    if len(file_paths_abs_1) == 0 and len(file_paths_abs_2) == 0:
        file_paths.append("Ingredients.docx")
    else:
        for file_path in file_paths_abs_1:
            file_paths.append(file_path)
        refs.extend(refs_1)
        for file_path in file_paths_abs_2:
            file_paths.append(file_path)
        refs.extend(refs_2)

        print(f"Titles are {file_titles_1} and {file_titles_2}")
            
    return file_paths, refs
  
async def analyze_harmful_ingredients(ingredient, assistant_id, client):
    
    is_ingredient_not_found_in_doc = False
    
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "A food product has the ingredient: " + ingredient + ". Is this ingredient safe to eat? The output must be in JSON format: {<ingredient_name>: <information from the document about why ingredient is harmful>}. If information about an ingredient is not found in the documents, the value for that ingredient must start with the prefix '(NOT FOUND IN DOCUMENT)' followed by the LLM's response based on its own knowledge.",
            }
        ]
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        include=["step_details.tool_calls[*].file_search.results[*].content"],
        tools=[{
        "type": "file_search",
        "file_search": {
            "max_num_results": 5
        }
        }]
    )
    
    
    ## List run steps to get step IDs
    #run_steps = client.beta.threads.runs.steps.list(
    #    thread_id=thread.id,
    #    run_id=run.id
    #)
    
    ## Initialize a list to store step IDs and their corresponding run steps
    #all_steps_info = []
    
    ## Iterate over each step in run_steps.data
    #for step in run_steps.data:  # Access each RunStep object
    #    step_id = step.id  # Get the step ID (use 'step_id' instead of 'id')
    
        ## Retrieve detailed information for each step using its ID
        #run_step_detail = client.beta.threads.runs.steps.retrieve(
        #    thread_id=thread.id,
        #    run_id=run.id,
        #    step_id=step_id,
        #    include=["step_details.tool_calls[*].file_search.results[*].content"]
        #)
    
        ## Append a tuple of (step_id, run_step_detail) to the list
        #all_steps_info.append((step_id, run_step_detail))
    
    ## Print all step IDs and their corresponding run steps
    #for step_id, run_step_detail in all_steps_info:
    #    print(f"Step ID: {step_id}")
    #    print(f"Run Step Detail: {run_step_detail}\n")
    
    # Polling loop to wait for a response in the thread
    messages = []
    max_retries = 10  # You can set a maximum retry limit
    retries = 0
    wait_time = 2  # Seconds to wait between retries

    while retries < max_retries:
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        if messages:  # If we receive any messages, break the loop
            break
        retries += 1
        time.sleep(wait_time)

    # Check if we got the message content
    if not messages:
        raise TimeoutError("Processing Ingredients : No messages were returned after polling.")
        
    message_content = messages[0].content[0].text
    annotations = message_content.annotations

    #citations = []

    #print(f"Length of annotations is {len(annotations)}")

    for index, annotation in enumerate(annotations):
      if file_citation := getattr(annotation, "file_citation", None):
          #cited_file = client.files.retrieve(file_citation.file_id)
          #citations.append(f"[{index}] {cited_file.filename}")
          message_content.value = message_content.value.replace(annotation.text, "")
  
    ingredients_not_found_in_doc = []        
    print(message_content.value)
    for key, value in json.loads(message_content.value.replace("```", "").replace("json", "")).items():
        if value.startswith("(NOT FOUND IN DOCUMENT)"):
            ingredients_not_found_in_doc.append(key)
            is_ingredient_not_found_in_doc = True
        print(f"Ingredients not found in database {','.join(ingredients_not_found_in_doc)}")
    
    harmful_ingredient_analysis = json.loads(message_content.value.replace("```", "").replace("json", "").replace("(NOT FOUND IN DOCUMENT) ", ""))
        
    harmful_ingredient_analysis_str = ""
    for key, value in harmful_ingredient_analysis.items():
      harmful_ingredient_analysis_str += f"{key}: {value}\n"
    return harmful_ingredient_analysis_str, is_ingredient_not_found_in_doc


def get_assistant_for_ingredient(ingredient, N=2, client, embeddings_titles_list):
  
    #Harmful Ingredients
    assistant2 = client.beta.assistants.create(
      name="Harmful Ingredients",
      instructions=f"You are an expert dietician. Use your knowledge base to answer questions about the ingredient {ingredient} in a food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )

    # Create a vector store
    vector_store2 = client.beta.vector_stores.create(
     name="Harmful Ingredients Vec",
     chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 400,  # Set your desired max chunk size
            "chunk_overlap_tokens": 200    # Set your desired overlap size
        }
    }
    )

    # Ready the files for upload to OpenAI.     
    file_paths, refs = get_files_with_ingredient_info(ingredient, embeddings_titles_list, N)
    #if file_paths[0] == "Ingredients.docx" and assistant_default_doc:
        #print(f"Using Ingredients.docx for analyzing ingredient {ingredient}")
    #    return assistant_default_doc, refs
        
    print(f"DEBUG : Creating vector store for files {file_paths} to analyze ingredient {ingredient}")
    
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch2 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store2.id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    print(file_batch2.status)
    print(file_batch2.file_counts)

    #harmful Ingredients
    assistant2 = client.beta.assistants.update(
      assistant_id=assistant2.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store2.id]}},
    )

    #if file_paths[0] == "../docs/Ingredients.docx" and assistant_default_doc is None:
    #    assistant_default_doc = assistant2
        
    return assistant2, refs

def analyze_processing_level(ingredients, assistant_id, client):
    
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Categorize food product that has following ingredients: " + ', '.join(ingredients) + " into Group A, Group B, or Group C based on the document. The output must only be the group category name (Group A, Group B, or Group C) alongwith the reason behind assigning that respective category to the product. If the group category cannot be determined, output 'NOT FOUND'.",
            }
        ]
    )
    
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id,
        include=["step_details.tool_calls[*].file_search.results[*].content"]
    )

    # Polling loop to wait for a response in the thread
    messages = []
    max_retries = 10  # You can set a maximum retry limit
    retries = 0
    wait_time = 2  # Seconds to wait between retries

    while retries < max_retries:
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        if messages:  # If we receive any messages, break the loop
            break
        retries += 1
        time.sleep(wait_time)

    # Check if we got the message content
    if not messages:
        raise TimeoutError("Processing Level : No messages were returned after polling.")
        
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    #citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, "")
        #if file_citation := getattr(annotation, "file_citation", None):
        #    cited_file = client.files.retrieve(file_citation.file_id)
        #    citations.append(f"[{index}] {cited_file.filename}")

    if debug_mode:
        print(message_content.value)
    processing_level_str = message_content.value
    return processing_level_str

@app.get("/api/processing_level-ingredient-analysis")
def get_ingredient_analysis(product_info_from_db):
        
    if product_info_from_db:
        brand_name = product_info_from_db.get("brandName", "")
        product_name = product_info_from_db.get("productName", "")
        ingredients_list = [ingredient["name"] for ingredient in product_info_from_db.get("ingredients", [])]

        processing_level = ""
        all_ingredient_analysis = ""
        claims_analysis = ""
        refs = []
        
        if len(ingredients_list) > 0:
            #Create client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            #Create assistant for processing level
            assistant_p, embeddings_titles_list = create_assistant_and_embeddings(client, ['../docs/embeddings.pkl', '../docs/embeddings_harvard.pkl'])
            #Create embeddings
  
            processing_level = analyze_processing_level(ingredients_list, assistant_p.id, client) if ingredients_list else ""
            for ingredient in ingredients_list:
                assistant_id_ingredient, refs_ingredient = get_assistant_for_ingredient(ingredient, 2, client, embeddings_titles_list)
                ingredient_analysis, is_ingredient_in_doc = analyze_harmful_ingredients(ingredient, assistant_id_ingredient.id, client)
                all_ingredient_analysis += ingredient_analysis + "\n"
                if is_ingredient_in_doc:
                    refs.extend(refs_ingredient)

        return refs, all_ingredient_analysis, processing_level
