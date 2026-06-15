import os
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import uuid
import time
from dotenv import load_dotenv

from utils.restricted_outputs import *
from utils.file_io import *
from utils.llm import *

def get_system_prompt_template_for_prompt_gen(category, bb=False):
    '''
    For a given category, retrieve the system prompt (as a string)
    '''
    file_path = Path(category_to_generating_prompt[category])
    if bb:
        assert category not in [1, 3], "Cannot do category 1 or 3 for bounding boxes"
        file_path = Path(str(file_path.with_suffix("")) + "_BB" + str(file_path.suffix))
    with open(file_path, "r") as f:
        prompt_template = f.read()
    return prompt_template 

def get_example_json_for_prompt_gen(category, bb=False):
    '''
    Given category for the type of prompt we want, retrieve the appropriate example json object
    '''
    file_path = Path(category_to_example[category])
    if bb:
        assert category not in [1, 3], "Cannot do category 1 or 3 for bounding boxes"
        file_path = Path(str(file_path.with_suffix("")) + "_BB" + file_path.suffix)
    
    return read_json(file_path)

def run_session(model_name, img_dir, img_name, category, out_dir, bb=False, toolbox=None):
    '''
    Create a single session, corresponding to a single image and a category (indicating type of prompt desired)
    If making bounding box variant (bb=True), provided toolbox of objects is used to guide prompt formation
    '''
    client = OpenAI()
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    if bb:
        assert toolbox is not None, "Must supply an object toolbox for bounding boxes"
        assert category not in [1, 3], "Cannot do category 1 or 3 for bounding boxes"

    # retrieve the system prompt template and example based on category
    prompt_examples = json_to_text(get_example_json_for_prompt_gen(category, bb))
    system_prompt_template = get_system_prompt_template_for_prompt_gen(category, bb)
    if (bb and toolbox is not None): 
        prompt_examples = f"Given the following object toolbox: {toolbox}\n\n" + prompt_examples
    
    system_prompt_filled = system_prompt_template.format(prompt_examples=prompt_examples)

    system_prompt = build_openai_schema(system_prompt_filled, None, is_system_prompt=True)

    full_img_path = img_dir / img_name

    # format user query
    if (not bb) or toolbox is None:
        user_instructions = "Generate a prompt for the following image."
    else:
        user_instructions = "Generate a prompt for the following image. Your object toolbox is: " + toolbox
    user_prompt = build_openai_schema(user_instructions, full_img_path, is_system_prompt=False)
    
    print(f"{system_prompt_filled=}")
    print(f"{user_instructions=}")

    for iteration in range(PROMPTS_PER_SESSION):
        # create PromptWrapper item
        new_id = str(uuid.uuid4())
        prompt_metadata = PromptMetadata(timestamp=timestamp, img=img_name, model=model_name, category=category, uuid=new_id)
        new_example_prompt = query_GPT(client, model_name, system_prompt, user_prompt, json_format=Prompt)
        prompt_wrapper = PromptWrapper(metadata=prompt_metadata, prompt=new_example_prompt)
        
        # save output
        output_dir = out_dir / Path(f"{img_name}/cat_{category}/{model_name}/")
        output_file_name = Path(new_id + ".json")

        output_dir.mkdir(parents=True, exist_ok=True)
        save_pydantic_to_file(prompt_wrapper, output_dir, output_file_name)

if __name__ == "__main__":
    load_dotenv()
    IS_BOUNDING_BOX = False
    
    img_dir = Path("./images/original/") # use original images
    
    if IS_BOUNDING_BOX:
        out_dir = Path("./generated_prompts/bb/")
        toolbox_map_dir = Path("./letter_to_class_map.json")
        toolbox_map = read_json(toolbox_map_dir)
        valid_categories = [0, 2]
    else:
        out_dir = Path("./generated_prompts/original/")
        valid_categories = list(range(4))
    
    # Loop through images and generate draft of prompts
    for img in os.listdir(img_dir):
        print("Generating prompt for image ", img)
        
        if IS_BOUNDING_BOX:
            # check that  we have enough objecst in toolbox
            toolbox = []
            num_valid_objects = 0
            for classification in toolbox_map[img]["classification"].values():
                if classification:
                    num_valid_objects += 1
                    toolbox.append(classification)
            if num_valid_objects < 2:
                print("Less than 2 valid objects in toolbox. Skipping image.")
                continue
        
        if (out_dir / Path(img)).exists():
            print(f"{out_dir / Path(img)} exists")
            continue
        
        print(f"{out_dir / Path(img)} doesn't already exist. Generating...")
        
        # Generate prompts across all desired categories for this image
        for category in valid_categories:
            start=time.perf_counter()
            run_session("gpt-5-mini", img_dir, img, category, out_dir, bb=IS_BOUNDING_BOX, toolbox=(','.join(toolbox) if IS_BOUNDING_BOX else None))
            end=time.perf_counter()

            print(f"Prompt generation time for image {img}: {end-start}")
        
