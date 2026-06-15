import json
import base64

def read_json(file_path):
    '''
    Given a file path, open the file and return a json object
    '''
    
    with open(file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file

def json_to_text(json_file, indent=None):
    '''
    Given a json object, convert it to string
    '''
    
    return json.dumps(json_file, indent=indent, ensure_ascii=False)

def convert_image_to_data_url(img_path):
    '''
    Encode image at img_path as data url
    '''
    
    file_extension = img_path.suffix.lstrip(".")
    map_extension = {"jpg": "jpeg", "png": "png"}
    print("Using file extension ", file_extension)
    with open(img_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/{map_extension[file_extension]};base64,{base64_image}"

def save_json_to_file(json_obj, file_dir, file_name):
    '''
    Given a json object, save to file_dir/file_name
    '''
    with open(file_dir / file_name, "w") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)

def save_pydantic_to_file(pydantic_obj, file_dir, file_name):
    '''
    Given a pydantic object, save to file_dir/file_name
    '''
    obj_as_json = pydantic_obj.model_dump()
    save_json_to_file(obj_as_json, file_dir, file_name)