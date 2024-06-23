import gradio as gr
import os
from PIL import ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import random
import numpy as np
import copy

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

model_path = "microsoft/Florence-2-large"

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="sdpa", trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model.to("cpu")

# Dictionary to store color mappings
color_map = {}

def get_color(object_type):
    if object_type not in color_map:
        # Generate a random color
        random.seed(object_type)
        color_map[object_type] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color_map[object_type]

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def draw_polygons(image, prediction, fill_mask=False):  
    """  
    Draws segmentation masks with polygons on an image.  
  
    Parameters:  
    - image_path: Path to the image file.  
    - prediction: Dictionary containing 'polygons' and 'labels' keys.  
                  'polygons' is a list of lists, each containing vertices of a polygon.  
                  'labels' is a list of labels corresponding to each polygon.  
    - fill_mask: Boolean indicating whether to fill the polygons with color.  
    """  
    draw = ImageDraw.Draw(image)  
    scale = 1  # Set up scale factor if needed (use 1 if not scaling)  
      
    # Iterate over polygons and labels  
    for polygons, label in zip(prediction['polygons'], prediction['labels']):  
        color = random.choice(colormap)  
        fill_color = random.choice(colormap) if fill_mask else None  
          
        for _polygon in polygons:  
            _polygon = np.array(_polygon).reshape(-1, 2)  
            if len(_polygon) < 3:  
                print('Invalid polygon:', _polygon)  
                continue  
              
            _polygon = (_polygon * scale).reshape(-1).tolist()  
              
            # Draw the polygon  
            if fill_mask:  
                draw.polygon(_polygon, outline=color, fill=fill_color)  
            else:  
                draw.polygon(_polygon, outline=color)  
              
            # Draw the label text  
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)  

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",
                    fill=color)

def detect_objects(image, task, text_input=None):
    if not task:
        return image, None
    
    task_prompt = f"<{task.upper()}>"
    results = run_example(task_prompt, image, text_input)
    
    draw = ImageDraw.Draw(image)
    # Load a font
    try:
        font = ImageFont.truetype(font="arial.ttf", size=100)
    except IOError:
        font = ImageFont.load_default(size=50)
    
    if task in ["od", "dense_region_caption", "region_proposal", "caption_to_phrase_grounding"]:
        for bbox, label in zip(results[f"<{task.upper()}>"]["bboxes"], results[f"<{task.upper()}>"]["labels"]):
            x0, y0, x1, y1 = bbox
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            if task == "od":
                color = get_color(label)
            else:
                color = "lightgreen"
                label = "" if task == "region_proposal" else label
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # Draw background rectangle for text
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_bbox, fill="white")
            
            draw.text((x0, y0), label, fill="black", font=font)
        
        return image, None
    elif task == "referring_expression_segmentation":
        output_image = copy.deepcopy(image)
        draw_polygons(output_image, results[f"<{task.upper()}>"], fill_mask=True)
        return output_image, None
    elif task == "ocr":
        return image, results[f"<{task.upper()}>"]
    elif task == "ocr_with_region":
        output_image = copy.deepcopy(image)
        draw_ocr_bboxes(output_image, results[f"<{task.upper()}>"])
        return output_image, None
    elif task == "open_vocabulary_detection":
        for bbox, label in zip(results[f"<{task.upper()}>"]["bboxes"], results[f"<{task.upper()}>"]["bboxes_labels"]):
            x0, y0, x1, y1 = bbox
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            color = get_color(label)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # Draw background rectangle for text
            text_bbox = draw.textbbox((x0, y0), label, font=font)
            draw.rectangle(text_bbox, fill="white")
            
            draw.text((x0, y0), label, fill="black", font=font)
        
        return image, None
    else:
        return image, results[f"<{task.upper()}>"]

# Task mapping for user-friendly names
task_mapping = {
    "Object Detection": "od",
    "OCR": "ocr",
    "Expression Segmentation": "referring_expression_segmentation",
    "Caption": "caption",
    "Detailed Caption": "detailed_caption",
    "More Detailed Caption": "more_detailed_caption",
    "Dense Region Caption": "dense_region_caption",
    "Region Proposal": "region_proposal",
    "OCR with Region": "ocr_with_region",
    "Caption to Phrase Grounding": "caption_to_phrase_grounding",
    "Open Vocabulary Detection": "open_vocabulary_detection"
}

# Sorted task names
sorted_task_names = [
    "Object Detection", 
    "OCR",
    "OCR with Region",  
    "Caption", 
    "Detailed Caption", 
    "More Detailed Caption",
    "Caption to Phrase Grounding",
    "Dense Region Caption", 
    "Region Proposal",
    "Expression Segmentation",
    "Open Vocabulary Detection"
    ]

# Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# 🪟 Microsoft Florence-2 Vision Model")
    gr.Markdown("Upload an image and select a task to perform.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            task_dropdown = gr.Dropdown(choices=sorted_task_names, label="Select Task", value=sorted_task_names[0])
            text_input = gr.Textbox(label="Text Input (Optional)", lines=1, placeholder="Enter text for referring expression segmentation or phrase grounding", visible=False)
            start_button = gr.Button("Start")
        with gr.Column():
            image_output = gr.Image(label="Output Image")
            caption_output = gr.Textbox(label="Caption Output", lines=5)

    def start_task(image, task_name, text):
        if not task_name:
            return image, None
        task = task_mapping[task_name]
        return detect_objects(image, task, text)

    def update_text_input(task_name):
        if task_name in ["Expression Segmentation", "Caption to Phrase Grounding", "Open Vocabulary Detection"]:
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
        
    def clear_text(text_input):
        return ""

    task_dropdown.change(fn=update_text_input, inputs=task_dropdown, outputs=text_input)
    start_button.click(fn=start_task, inputs=[image_input, task_dropdown, text_input], outputs=[image_output, caption_output]).then(
        clear_text, inputs=[text_input], outputs=[text_input]
    )

demo.launch()