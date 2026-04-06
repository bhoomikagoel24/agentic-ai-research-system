from openai import OpenAI
from research_agent_system.config.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
