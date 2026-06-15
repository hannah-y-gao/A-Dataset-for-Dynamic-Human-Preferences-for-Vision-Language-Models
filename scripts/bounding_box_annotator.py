from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import cv2
import os
from openai import OpenAI
from dotenv import load_dotenv

from utils.llm import *
from utils.file_io import *
from utils.restricted_outputs import *

def run_yolo(img_path: Path, out_dir: Path):
    '''
    Runs Yolo model on image at img_path to generate bounding boxes  
    Saves image to out_dir and returns mappings from bounding box letters to objects
    '''
    
    model = YOLO("yolo26n.pt")
    
    cls_to_label_map = model.names
    box_labels = ["A", "B", "C", "D", "E"] # allow up to 5 boxed objects
    color_palette_indices = [0, 6, 9, 13, 18] # select colors from palette
    colors = Colors()
    
    file_path = img_path.resolve()
    image = cv2.imread(file_path)
    annotator = Annotator(image, line_width=2, font_size=1)

    # predict bounding moxes
    results = model(str(file_path), imgsz=640, half=True, save=False, device="cpu")      
    
    # filter out duplicate classes
    duplicated_classes = set()
    seen_classes = set()
    for i in range(len(results[0])):
        class_idx = results[0].boxes.cls[i].item()
        if class_idx in seen_classes:
            duplicated_classes.add(class_idx)
        else:
            seen_classes.add(class_idx)
    
    # process letter to class mappings
    letter_to_class_map = dict()
    for i in range(len(results[0])): 
        class_idx = results[0].boxes.cls[i].item()
        if class_idx in duplicated_classes:
            continue
        true_class = cls_to_label_map[results[0].boxes.cls[i].item()]
        if true_class == "person": # remove people
            continue
        letter = box_labels[len(letter_to_class_map)]
        annotator.box_label(box=(results[0].boxes.xyxy)[i], label=letter, color=colors(color_palette_indices[len(letter_to_class_map)], True))
        letter_to_class_map[letter] = true_class
        
        # cap at 5 boxed objects
        if len(letter_to_class_map) >= len(box_labels):
            break
        
    out_file_path = out_dir / f"{img_path.stem}_BB.jpg"
    annotator.save(str(out_file_path))
    print("BB Image saved to " + str(out_file_path))
    
    return letter_to_class_map

def generate_bounding_boxes(img_source_dir: Path, img_target_dir: Path):
    '''
    Generates images with bounding box annotations for all images in img_dir
    Return a mapping for each image from letters to classes
    '''
    img_to_letter_to_class_map = dict()
    for img in os.listdir(img_source_dir):
        full_img_path = img_dir / img
        letter_to_class_map = run_yolo(full_img_path, img_target_dir)   
        
        img_to_letter_to_class_map[img] = letter_to_class_map
    
    return img_to_letter_to_class_map
    
if __name__ == "__main__":
    load_dotenv()
    
    # generate bounding boxes for images
    img_dir = Path("./images/original/")
    bb_img_dir = Path("./images/bb/")
    bb_mapping = generate_bounding_boxes(img_dir, bb_img_dir)
    
    # update bounding box to object mappings using LLM
    model_name = "gpt-5-mini"
    
    client = OpenAI()
    modified_bb_mapping = dict()
    
    # loop through all images and add bounding boxes
    for img_name, letter_to_class_map in bb_mapping.items():
        img_path = img_dir / Path(img_name)
        assert img_path.exists(), f"file path {img_path} doesn't exist"
        
        # revise mapping with a VLM
        system_prompt_file = "./system_prompts/bounding_box_annotation/mapping_revision.txt"
        with open(system_prompt_file) as f:
            system_prompt_txt = f.read()
        user_prompt_txt = json_to_text(letter_to_class_map)
        
        system_prompt = build_openai_schema(system_prompt_txt, None, True)
        user_prompt = build_openai_schema(user_prompt_txt, img_path, False)        
        LLM_prompt = query_GPT(client, model_name, system_prompt, user_prompt, BoundingBoxMappingRevision)
        
        if LLM_prompt is not None:
            modified_bb_mapping[img_name] = LLM_prompt.model_dump()
        else:
            raise Exception("LLM_prompt was None")
        
    save_json_to_file(modified_bb_mapping, Path("./"), Path("letter_to_class_map.json"))
        
        
        
        
        