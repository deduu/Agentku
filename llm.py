import openai
from settings import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_action_from_user_input(user_input, parsed_screen):
    prompt = f"""
You are an AI agent. The user said: "{user_input}"
Here is the current GUI layout:\n{parsed_screen}

Decide what to do next. Reply in JSON like:
{{
  "action": "click" / "type",
  "target_content": "...",
  "value": "...",  # only if action is "type"
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    import json
    return json.loads(response['choices'][0]['message']['content'])
