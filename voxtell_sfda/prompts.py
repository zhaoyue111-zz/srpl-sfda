DEFAULT_PROMPTS = [
    "spleen",
    "right kidney",
    "left kidney",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "inferior vena cava",
    "duodenum",
    "pancreas",
    "esophagus",
]


def print_label_mapping(prompts):
    print("Prompt/label mapping:")
    for label_id, prompt in enumerate(prompts, start=1):
        print(f"  {label_id:2d}: {prompt}")
