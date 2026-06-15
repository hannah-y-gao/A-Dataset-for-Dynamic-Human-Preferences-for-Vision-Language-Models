from generate_prompts import *
from openai import OpenAI
from dotenv import load_dotenv


def get_revision_issues(category, bb=False):
    '''
    Given a category, returns the revision issues type to description dictionary
    '''
    if bb:
        assert category not in [1, 3], "Cannot do category 1 or 3 for bounding boxes"
    return {k:v for (k, v) in revision_issues.items() if k not in category_to_revision_exclusions[category]}

def request_review(client, model_name, system_prompt_txt, user_prompt_json, img_path, relevant_issues, bb=False):
    '''
    Returns the Issue object for a user_prompt_json
    '''
    user_prompt_txt = json_to_text(user_prompt_json)
    system_prompt = build_openai_schema(system_prompt_txt, None, True)
    user_prompt = build_openai_schema(user_prompt_txt, img_path, False)
    
    print("\n\n-------------System prompt for requesting review-------------\n")
    print(system_prompt)
    print(img_path)
    # print("\n\n-------------User prompt for requesting review-------------\n")
    # print(user_prompt)
    with open("system prompt for requesting review", "w") as f:
        f.write(system_prompt_txt)
    with open("user prompt for requesting review", "w") as f:
        f.write(user_prompt_txt)
        
    
    review_response = query_GPT(client, model_name, system_prompt, user_prompt, Issue)
    issue_verdict = review_response.issue

    if issue_verdict != "none" and issue_verdict not in relevant_issues:
        raise Exception(f"issue verdict {issue_verdict} not valid")
    
    return review_response

def revise_prompt(client, model_name, prompt_issue, user_prompt_json, img_path, bb=False):
    '''
    Given a user_prompt_json and an Issue object, request revision of prompt
    '''
    # get proper system prompt for revising for the particular issue
    issue_type = " ".join((prompt_issue.issue).split("_"))
    if issue_type != 'none':
        system_prompt_file = "system_prompts/revision_pipeline/revisor_prompt_BB.txt" if bb else "system_prompts/revision_pipeline/revisor_prompt.txt"
        suggested_fix = revision_suggested_fixes[prompt_issue.issue]
        with open(system_prompt_file) as f:
            system_prompt_txt = (f.read()).format(issue_type=issue_type, issue_description=prompt_issue.analysis, suggested_fix=suggested_fix)
    else:
        system_prompt_file = "system_prompts/revision_pipeline/responder_prompt_BB.txt" if bb else "system_prompts/revision_pipeline/responder_prompt.txt"
        with open(system_prompt_file) as f:
            system_prompt_txt = f.read()
    
    system_prompt = build_openai_schema(system_prompt_txt, None, True)
    user_prompt_txt = json_to_text(user_prompt_json)
    user_prompt = build_openai_schema(user_prompt_txt, img_path, False)

    print("\n\n-------------System prompt for revising prompt-------------\n")
    print(system_prompt)
    
    print("\n\n-------------User prompt for revising prompt-------------\n")
    print(user_prompt)
    
    if prompt_issue.issue != 'none':
        prompt_revision = query_GPT(client, model_name, system_prompt, user_prompt, PromptRevision)
    else:
        # if there is no issue, put "answer" under "proposed_fix" and return same prompt w/ no revision
        answer = query_GPT(client, model_name, system_prompt, user_prompt, Answer).answer
        prompt_revision = PromptRevision(proposed_fix=answer, revised_prompt=Prompt(**user_prompt_json))
    return prompt_revision


def generate_revision(model_name, file_dir, file_name, img_dir, bb=False):
    '''
    Given the file directory and file name, generate a revision using the LLM specified
    Output revision in a subfolder of file_dir called revisions
    '''
    client = OpenAI()
    
    original_prompt_wrapper = read_json(file_dir / file_name)
    metadata = original_prompt_wrapper["metadata"]

    full_img_path = img_dir / metadata["img"]
    uuid = metadata["uuid"]
    category = metadata["category"]
    original_prompt = original_prompt_wrapper["prompt"]

    revision_prompt_path = "system_prompts/revision_pipeline/reviewer_prompt_BB.txt" if bb \
        else "system_prompts/revision_pipeline/reviewer_prompt.txt"
    
    with open(revision_prompt_path, "r") as f:
        reviewer_prompt_template = f.read()

    # certain categories of data may not require all types of revisions
    filtered_revision_issues = get_revision_issues(category, bb)
    revision_issue_descriptions = "\n".join([f"{k}: {v}" for (k, v) in filtered_revision_issues.items()])
    
    system_prompt_txt = reviewer_prompt_template.format(revision_issue_descriptions=revision_issue_descriptions, revision_issue_types= ", ".join(list(filtered_revision_issues.keys())))

    issue_verdict = ""
    revision_history = RevisionSession(img=metadata["img"], model=model_name, category=category, uuid=uuid, original_prompt=original_prompt, revised_prompts=[])
    updated_user_prompt_json = original_prompt # modified version of user prompt
    
    while issue_verdict != "none":
        # retrieve Issue (analysis, issue) from reviewing this prompt
        prompt_issue = request_review(client, model_name, system_prompt_txt, updated_user_prompt_json, full_img_path, filtered_revision_issues.keys(), bb)
        issue_verdict = prompt_issue.issue

        # Retrieve a PromptRevision
        new_prompt_revision = revise_prompt(client, model_name, prompt_issue, updated_user_prompt_json, full_img_path, bb) 

         # update the latest user prompt
        updated_user_prompt_json = new_prompt_revision.revised_prompt.model_dump()

        # create new Revision entry from the PromptRevision and add to RevisionSession object
        new_revision_entry = RevisionEntry(issue=prompt_issue, prompt_revision=new_prompt_revision)
        revision_history.revised_prompts.append(new_revision_entry)
       

    output_dir = file_dir / "revisions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    i=0
    save_path = Path(Path(file_name).stem + f"_revised{i}" + Path(file_name).suffix)
    while((output_dir / save_path).exists()):
        i += 1
        save_path = Path(Path(file_name).stem + f"_revised{i}" + Path(file_name).suffix)
    save_pydantic_to_file(revision_history, output_dir, save_path)

if __name__ == "__main__":
    load_dotenv()
    IS_BOUNDING_BOX = True
    
    if IS_BOUNDING_BOX:
        generated_prompts_dir = Path("./generated_prompts/bb/")
    else:
        generated_prompts_dir = Path("./generated_prompts/original/")
    generated_prompts = list(generated_prompts_dir.rglob('*.json'))
    
    # for each prompt draft, generate revisions
    for obj_path in generated_prompts:
        prompt_dir = obj_path.parent
        prompt_file = obj_path.name
        
        if "_revised" in prompt_file:
            continue
        if (prompt_dir / "revisions").exists():
            print(f"Skipping {prompt_dir}.")
            continue
    
        generate_revision("gpt-5-mini", prompt_dir, prompt_file, Path("./images/original/"))
        
    