from transformers import AutoModelForObjectDetection, AutoProcessor
import torch
from PIL import Image

model = AutoModelForObjectDetection.from_pretrained("microsoft/OmniParser-v2.0")
processor = AutoProcessor.from_pretrained("microsoft/OmniParser-v2.0")

def parse_screen(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    screen_elements = processor.post_process_object_detection(outputs, target_sizes=[image.size], threshold=0.1)[0]

    parsed = []
    for box, score, label in zip(screen_elements["boxes"], screen_elements["scores"], screen_elements["labels"]):
        bbox = [round(v.item(), 3) for v in box]
        parsed.append({
            "label": processor.tokenizer.decode([label]),
            "score": round(score.item(), 3),
            "bbox": bbox
        })
    return parsed
