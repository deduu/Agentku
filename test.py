from openai import OpenAI
from settings import settings

print(settings.OPENAI_API_KEY)
client = OpenAI(api_key=settings.OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Write a one-sentence bedtime story about a unicorn."
    }]
)

print(completion.choices[0].message.content)