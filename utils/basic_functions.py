import os,requests,ast,torch
import gradio as gr
import datetime as dt
import google.generativeai as genai
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForVision2Seq,LlavaForConditionalGeneration

# function for pulling secrets from local repositories
def get_secret(secret_key):
    if not os.getenv(secret_key): # usually used in other repos when github actions is utilized
        env_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\..','.gitignore\.env'))
        load_dotenv(dotenv_path=env_path)
    
    value = os.getenv(secret_key)
    print(''.join(['*']*len(value)))
    if value is None:
        ValueError(f"Secret '{secret_key}' not found.")
    
    return value

# download an image when when provided a url
def get_image(url):
    # 1. Fetch the image and download the image
    try:
        response = requests.get(url,stream=True)
        response.raise_for_status()
        content = response.content

        #with open(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\download\\{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg', 'wb') as f:
        #    f.write(content)
    except requests.exceptions.RequestException as e:
        print(f'Error downloading image: {e}')
        exit()
    except IOError as e:
        print(f'Error saving image file: {e}')
        exit()
    return Image.open(BytesIO(content)).convert("RGB")

def load_model(model_name):
    if 'llava' in model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name,use_fast=True)
    return processor,model

def gemini_identify_id(url,system_prompt):
    # 2. Function to process image with Gemini Pro Vision
    try:
        image = get_image(url)
        
        genai.configure(api_key=get_secret('GEMINI_API'))

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([system_prompt, image])
        response_text = response.text
        if not response_text:
            print('Could not find an ID number')
            exit()
        print(response_text)
    
    except Exception as e:
        return f"Error processing image: {str(e)}",None
    
    draw = ImageDraw.Draw(image)
    draw.rectangle(ast.literal_eval(response_text),outline='yellow',width=5)
    image.save(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\download\\{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')

# Huggingface repo usage
def huggingface_detect_id_box(model_name,url):
    try:
        image = get_image(url)

        system_prompt = f"""
        You are an AI document processing assistant. Analyze the provided image. Identify the ID number in the document.
        This is usually identified in a location outside of the main content on the document, and usually on the bottom
        right or left of the document. The rotation of the number may differ based on images. Furthermore the ID number
        is usually a string of numbers, around 9 number characters in length. Could possibly have alphabetic characters
        as well but that looks to be rare. The output should only be a string in the format [x0,y0,x1,y1], and the
        values should fit into the image size which is {image.size}.
        """

        processor,model=load_model(model_name)
        inputs = processor(image,text=system_prompt,return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs)
        
        response_text = processor.batch_decode(output,skip_special_tokens=True)[0]
        print(response_text)
        try:
            bbox = ast.literal_eval(response_text)
        except Exception as e:
            print(f"Error parsing bounding box response: {str(e)}")
            return None
        
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox,outline="red",width=5)
        #image.save(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\download\\{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
        return image
    except Exception as e:
        print(f"Error loading model or processing image: {str(e)}")
        return None