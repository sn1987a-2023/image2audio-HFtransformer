
from dotenv import find_dotenv, load_dotenv

from transformers import pipeline

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
import requests

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# img2text
def img2text(url):
    image_to_text = pipeline("image-to-text", model = "Salesforce/blip-image-captioning-base", max_new_tokens=512)
    text = image_to_text(url)[0]["generated_text"]
    return text
                              

# #llm to generate stories
def generate_story(senario):
    template= """
    You are a story teller;
    you can generate a short story based on a simple narrative, the story should be 3-5 sentences long and no more than 50 words;
    
    CONTEXT: {senario}
    STORY:
    
    """
    
    prompt = PromptTemplate(template = template, input_variables = ["senario"])
    
    story_llm = LLMChain(llm=ChatOpenAI(
        model_name = "gpt-3.5-turbo", 
        temperature=1), 
        prompt = prompt,
        verbose = True)
    
    story = story_llm.predict(senario=senario)

    return story

# # text2speech
def text2speech(text):

    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
      
        return response
   
    payload = {
        "inputs": text
    }

    response = query(payload=payload)
    
    with open("audio.flac", "wb") as file:

        file.write(response.content)


def main():

    senario = img2text("./image.png")

    story = generate_story(senario)

    text2speech(story)

if __name__ == "__main__":
    main()


