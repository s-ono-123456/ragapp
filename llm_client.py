import openai
import os

def get_gpt_response(prompt):
    client = openai.Client()
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
