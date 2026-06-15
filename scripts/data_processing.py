from pathlib import Path
from utils import *
import random
import re

from utils.file_io import *

def replace_object_with_bb(answer_choices, label_to_object_map):
    '''
    Replace objects in answer choices (dictionary) with their corresponding bounding box \
        label found in label_to_object_map
    '''
    object_to_label_map = {}
    # filter out duplicates & empty entries in map
    for label, obj in label_to_object_map.items():
        if not obj: # if classification is ''
            continue
        if obj in object_to_label_map:
            raise Exception(f"Duplicate object in mapping: {label_to_object_map}")
        object_to_label_map[obj] = "object in bounding box "+ label
    
    # make object substitutions
    new_answer_choices = {}
    for letter, answer in answer_choices.items():
        new_answer = re.sub(r'<object>(.*?)</object>', lambda matchObj : object_to_label_map.get(matchObj.group(1), matchObj.group(1)), answer)
        print(f"{answer=}")
        print(f"{new_answer=}")
        new_answer_choices[letter] = new_answer
    return new_answer_choices
    
    
def consolidate_generated_examples(search_dir, output_dir, is_bb=False, json_mapping_path=None):
    '''
    For each prompt in search_dir, extracts the final prompt from revisions, shuffles answer choices \
        and replaces objects with bounding box labels    
    '''
    total_examples = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_folder = "revisions"
    
    if is_bb and json_mapping_path is not None:
        bb_mapping = read_json(json_mapping_path)
    
    choices = ["a", "b", "c", "d"]
    # search for prompt json files
    for matching_folder in search_dir.rglob(target_folder):  
        print("matching folder found")
        if not matching_folder.is_dir():
            continue
        for json_file in matching_folder.rglob("*.json"):
            print("-"*60)
            raw_json = read_json(json_file)
            
            print("consolidate image " + raw_json["img"])
            
            category_dir = output_dir / raw_json["img"]
            category_dir.mkdir(exist_ok=True)
            
            prompt = raw_json["revised_prompts"][-1]["prompt_revision"]["revised_prompt"]
            original_answer = raw_json["revised_prompts"][-1]["prompt_revision"]["proposed_fix"]
            original_choices = sorted(prompt["choices"].items())
            
            # shuffle answer choices
            labeled_choices = [(i, choice) for i, (letter, choice) in enumerate(original_choices)]
            random.shuffle(labeled_choices)
            shuffled_choices = {}
            new_answer = None
            for new_idx, (old_idx, choice) in enumerate(labeled_choices):
                new_letter = choices[new_idx]
                old_letter = choices[old_idx]
                if (original_answer == old_letter):
                    new_answer = new_letter
                shuffled_choices[new_letter] = choice
            
            if new_answer is None:
                raise Exception("No new answer assigned")
            
            # make bounding box label substitutions
            if is_bb and json_mapping_path is not None:
                prompt['choices'] = replace_object_with_bb(shuffled_choices, bb_mapping[raw_json["img"]]['classification'])
            else:
                prompt['choices'] = shuffled_choices
            
            # save new processed json
            filtered_json = {"img": raw_json["img"], "model": raw_json["model"], "category": raw_json["category"], "uuid": raw_json["uuid"], "prompt": prompt, "answer": new_answer}
            save_json_to_file(filtered_json, category_dir, json_file.name)
            total_examples += 1
    
    print(f"Consolidated {total_examples} examples")
    return total_examples
            
def generate_form_assignments(search_dir, output_dir, total_examples, num_groups, is_bb=False):
    '''
    Go through every image and assign categories 0-v to indices i+0 to i+3, incrementing i at each image \
        (v is number of variations of prompt we have, typically 4 but 2 for bounding box)
    '''
    v = 2 if is_bb else 4
    
    assert total_examples%v == 0, f"we don't have {v} categories per image"
    
    output_dir.mkdir(exist_ok=True)
    group_num = 0
    group_to_uuids = {(i, j): set() for i in range(num_groups) for j in range(v)} # maps (group number, category number) to set of items
    
    for image_folder in search_dir.iterdir():
        
        categories_found = set()
        
        # distribute categories for a single image in a round across the groups
        for json_file in image_folder.rglob("*.json"):
            data_sample = read_json(json_file)
            category = int(data_sample["category"])
            
            shift = int(category==2) if is_bb else category
            
            shifted_group_num = (group_num+shift)%num_groups
            assign_to_folder = output_dir / f"group_{shifted_group_num}/"
            assign_to_folder.mkdir(exist_ok=True)
            save_json_to_file(data_sample, assign_to_folder, json_file.name)
            
            group_to_uuids[(shifted_group_num, shift)].add(data_sample["uuid"])
            categories_found.add(category)

        group_num += 1
        group_num %= num_groups
        
        if (len(categories_found) != v):
            print("ERROR: " + f" image {image_folder} has categories {categories_found}")
    
    # save stats of generated prompts
    stats_file = output_dir / "stats.txt"
    with open(stats_file, "w") as f:
        f.write("Generated the following groups: \n\n")
        for k, v in group_to_uuids.items():
            f.write(f"Form number {k[0]}, category {k[1]}: {len(v)} examples\n")
            
if __name__ == "__main__":
    IS_BOUNDING_BOX = True
    
    num_groups = 1
    
    if IS_BOUNDING_BOX:
        in_dir = Path("./generated_prompts/bb/")
        out_dir = Path("./processed_generated_examples/bb/")
        json_mapping_path = Path('./letter_to_class_map.json')
    else:
        in_dir = Path("./generated_prompts/original/")
        out_dir = Path("./processed_generated_examples/original/")
        json_mapping_path = None
        
    total_num_examples = consolidate_generated_examples(in_dir, out_dir, is_bb=IS_BOUNDING_BOX, json_mapping_path=json_mapping_path)
    generate_form_assignments(search_dir=out_dir, output_dir=out_dir / Path(f"./partitioned_{num_groups}_groups/"), total_examples=total_num_examples, num_groups=num_groups, is_bb = IS_BOUNDING_BOX)