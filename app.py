import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './hoo01_robot/'
os.system(f'git clone https://code.openxlab.org.cn/hoo01/hoo01_robot.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system("pip install sentencepiece")
os.system("pip install einops")
os.system("pip install transformers")
os.system("pip install --upgrade gradio")
'''from lmdeploy import pipeline, TurbomindEngineConfig
backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2) 

pipe = pipeline(base_path, backend_config=backend_config)

def model(image, text):
    response = pipe((text, image)).text
    return [(text, response)]

demo = gr.Interface(fn=model, inputs=[gr.Textbox(),], outputs=gr.Chatbot())
demo.launch()  '''

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True)

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="hoo01_robot",
                description="""
我是hoo01的robot.  
                 """,
                 ).queue(1).launch()
