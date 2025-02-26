import os,json,sys
import gradio as gr
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.basic_functions import get_image,gemini_identify_id,huggingface_detect_id_box

def main():

    demo = gr.Interface(
        fn=huggingface_detect_id_box,
        inputs=[
            gr.Textbox(label='Huggingface Model Name',placeholder='llava-hf/llava-1.5-7b-hf'),
            gr.Textbox(label="Image URL",placeholder="https://cumberland.isis.vanderbilt.edu/stefan/rvlcdip/test_advertisement_127238fe-8352-49ed-8c5f-11488a154dd2.jpg")
        ],
        outputs=gr.Image(label="Image with ID bounding box"),
        title="Huggingface LLM for ID Number Detection"
    )

    demo.launch()

if __name__ == "__main__":
    main()