import logging
import traceback
import sys
from functools import wraps
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import torch
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(debug = True)

def create_assistant(client):
    assistant3 = client.beta.assistants.create(
      name="Misleading Claims",
      instructions="You are an expert dietician. Use your knowledge base to answer questions about the misleading claims about food product.",
      model="gpt-4o",
      tools=[{"type": "file_search"}],
      temperature=0,
      top_p = 0.85
      )

    # Create a vector store
    vector_store3 = client.beta.vector_stores.create(name="Misleading Claims Vec")
    
    # Ready the files for upload to OpenAI
    file_paths = ["docs/MisLeading_Claims.docx"]
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch3 = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id=vector_store3.id, files=file_streams
    )

    #Misleading Claims
    assistant3 = client.beta.assistants.update(
      assistant_id=assistant3.id,
      tool_resources={"file_search": {"vector_store_ids": [vector_store3.id]}},
    )

    return assistant3
  
def analyze_harmful_ingredients(ingredient, assistant_id, client):
    
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

# Define the request body using a simple BaseModel (without complex pydantic models if not needed)
class IngredientAnalysisRequest(BaseModel):
    product_info_from_db: dict
    
@app.post("/api/processing_level-ingredient-analysis")
def get_ingredient_analysis(request: IngredientAnalysisRequest):
    product_info_from_db = request.product_info_from_db
        
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
            assistant_p, embeddings_titles_list = create_assistant_and_embeddings(client, ['docs/embeddings.pkl', 'docs/embeddings_harvard.pkl'])
            #Create embeddings
  
            processing_level = analyze_processing_level(ingredients_list, assistant_p.id, client) if ingredients_list else ""
            for ingredient in ingredients_list:
                assistant_id_ingredient, refs_ingredient = get_assistant_for_ingredient(ingredient, client, embeddings_titles_list, 2)
                ingredient_analysis, is_ingredient_in_doc = analyze_harmful_ingredients(ingredient, assistant_id_ingredient.id, client)
                all_ingredient_analysis += ingredient_analysis + "\n"
                if is_ingredient_in_doc:
                    refs.extend(refs_ingredient)

        return {'refs' : refs, 'all_ingredient_analysis' : all_ingredient_analysis, 'processing_level' : processing_level}
