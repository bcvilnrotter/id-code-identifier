import os,requests,ast,torch,re,anthropic
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'llava' in model_name:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
    print(f"model: {model}")
    processor = AutoProcessor.from_pretrained(model_name,use_fast=True)
    print(f"processor: {processor}")
    return processor,model

def request_manager(model_name,url):
    image = get_image(url)
    print(f"image: {image}")

    system_prompt = f"""
    You are an AI document processing assistant. Analyze the provided image. Identify the ID number in the document.
    This is usually identified in a location outside of the main content on the document, and usually on the bottom
    right or left of the document. The rotation of the number may differ based on images. Furthermore the ID number
    is usually a string of numbers, around 9 number characters in length. Could possibly have alphabetic characters
    as well but that looks to be rare. The output should only be a string in the format [x0,y0,x1,y1], and the
    values should fit into the image size which is {image.size}.
    """
    print(f"system_prompt: {system_prompt}")

    return_packet = [None,None]
    
    if 'gemini' in model_name:
        return_packet = gemini_identify_id(model_name,image,system_prompt)
    elif 'llava' in model_name:
        return_packet = huggingface_llava_15_7b_hf(model_name,image,system_prompt)
    elif 'claude' in model_name:
        return_packet = anthropic_identify_id(model_name,system_prompt,url)
    return return_packet

def anthropic_identify_id(model_name,system_prompt,url):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model = model_name,
        max_tokens = 1024,
        messages = [
            {
                "role":"user",
                "content": [
                    {
                        "type":"image",
                        "source":{
                            "type":"url",
                            'url':url
                        },
                    },
                    {
                        'type':'text',
                        'text':system_prompt
                    }
                ],
            }
        ],
    )
    print(message)
    return [None,None]

def gemini_identify_id(model_name,image,system_prompt):
    # 2. Function to process image with Gemini Pro Vision
    try:
        genai.configure(api_key=get_secret('GEMINI_API'))
        print(f"genai: {genai}")
        model = genai.GenerativeModel("gemini-2.0-flash")
        print(f"model: {model}")
        response = model.generate_content([system_prompt, image])
        print(f"response: {response}")
        response_text = response.text
        print(f"response: {response_text}")
        if not response_text:
            print('Could not find an ID number')
            return [image,'no response was received']
    
    except Exception as e:
        return [image,f"Error processing image: {str(e)}"]
    
    draw = ImageDraw.Draw(image)
    print(f"draw: {draw}")
    draw.rectangle(ast.literal_eval(response_text),outline='yellow',width=5)
    #image.save(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\download\\{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
    return [image,response_text]

# Huggingface repo usage
def huggingface_llava_15_7b_hf(model_name,image,system_prompt):
    try:
        #image = get_image(url)
        processor,model=load_model(model_name)
        conversation = [
            {
                "role":"user",
                "content":[
                    {"type":"text","text":system_prompt},
                    {"type":"image"},
                ],
            },
        ]
        print(f"conversation: {conversation}")
        
        prompt = processor.apply_chat_template(conversation,add_generation_prompt=True)
        print(f"prompt: {prompt}")
        
        inputs = processor(images=image,text=prompt,return_tensors="pt").to(model.device)
        print(f"inputs: {inputs}")

        """
        with torch.no_grad():
            output = model.generate(**inputs)
        
        response_text = processor.batch_decode(output,skip_special_tokens=True)[0]
        print(response_text)
        try:
            bbox = ast.literal_eval(response_text)
        except Exception as e:
            print(f"Error parsing bounding box response: {str(e)}")
            return None
        """

        output = model.generate(**inputs,max_new_tokens=200,do_sample=False)
        print(f"output: {output}")
        
        response_string = processor.decode(output[0][2:],skip_special_tokens=True)
        print(f"response_string: {response_string}")

        match = re.search(r"ASSISTANT: \[(.*?)\]",response_string)
        if not match:
            return [image,"no match found"]
        bbox = [image.size[0],image.size[1],image.size[0],image.size[1]]*ast.literal_eval([match.group(1)])
        print(f"bbox: {bbox}")

        
        draw = ImageDraw.Draw(image)
        print(f"draw: {draw}")
        
        draw.rectangle(bbox,outline="red",width=5)
        print(f"image: {image}")
        
        #image.save(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\\download\\{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg')
        return [image,bbox]
    except Exception as e:
        print(f"Error loading model or processing image: {str(e)}")
        return [image,"an error occurred processing request"]