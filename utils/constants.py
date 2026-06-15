PROMPTS_PER_SESSION = 1

category_to_tags = {
    0: ["easy", "low_dependence"],
    1: ["easy", "high_dependence"],
    2: ["hard", "low_dependence"],
    3: ["hard", "high_dependence"],
}

category_to_generating_prompt = {
    0: "./system_prompts/generation/0.txt",
    1: "./system_prompts/generation/1.txt",
    2: "./system_prompts/generation/2.txt",
    3: "./system_prompts/generation/3.txt"
}

category_to_example = {
    0: "./system_prompts/generation/examples/0.json",
    1: "./system_prompts/generation/examples/1.json",
    2: "./system_prompts/generation/examples/2.json",
    3: "./system_prompts/generation/examples/3.json",
}

category_to_revision_exclusions = {
    0: ["low_image_dependency"],
    1: [],
    2: ["low_image_dependency"],
    3: [],
}
revision_issues = {
    "wrong_scene_description": "The image description under the 'scene' key includes incorrect elements such as hallucinations.",
    "poor_preference_construction": "The human preference is hard to understand or does not suit the image well.",
    "no_correct_choices": "None of the answer choices are correct.",
    "multiple_correct_choices": "More than one of the answer choices are correct",
    "low_image_dependency": "The correct answer choice can be easily deduced from the preference description and answer choices themselves, without referencing the image.",
    "non_actionable": "An answer choice does not start with a verb (i.e. is non-actionable)."
}

revision_suggested_fixes = {
    "wrong_scene_description": "Replace the scene description with a concise and accurate scene description.",
    "poor_preference_construction": "Replace the human preference with a new one that is clearer and relevant to the image.",
    "no_correct_choices": "Replace one of the incorrect answer choices with a correct answer choice. The answer choice should be actionable (i.e. start with a verb) and should be related to the image and stated preference.",
    "multiple_correct_choices": "Pick a correct answer choice to keep. Replace all but one of the correct answer choices with an incorrect answer choice, such as an action that contradicts the stated human preference.",
    "low_image_dependency": "Replace the answer choices that have low image dependency with a slightly more abstracted version of the answer choice, such as describing locations in relative terms and avoiding directly re-stating phrases from the human preference.",
    "non_actionable": "Replace the non-actionable answer choice with an actionable version of the answer choice. Maintain correctness of the answer."
}