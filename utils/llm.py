from utils.file_io import convert_image_to_data_url

def query_GPT(client, model_name, system_component, user_component, json_format, custom_reasoning=None):
    '''
    Given the formatted schemas (both lists) for system & user components, combine and query GPT
    '''
    
    full_input = system_component + user_component

    if json_format is not None:
        response = client.responses.parse(
            model=model_name,
            input=full_input,
            text_format=json_format,
        )
        verdict = response.output_parsed
    else:
        if custom_reasoning is None:
            response = client.responses.create(
                model=model_name,
                input=full_input,
            )
        else:
            response = client.responses.create(
                model=model_name,
                input=full_input,
                reasoning={'effort': 'medium'}
            )
        verdict = response.output_text
    return verdict

def build_openai_schema(text, img_path, is_system_prompt):
    '''
    Given text and image, return formatted prompt for OpenAI model calls
    '''
    query_content = [
                        {
                            "type": "input_text", "text": text
                        }
                    ] + (
                        [] if img_path is None else \
                            [{"type": "input_image", "image_url": convert_image_to_data_url(img_path), "detail": "high"}])
    formatted_system_prompt = [
        {
            "role": "system" if is_system_prompt else "user",
            "content": query_content,
        },
    ]
    
    return formatted_system_prompt