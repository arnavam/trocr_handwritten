import json
import argparse
from os.path import join
import os
import sys

from jinja2 import Template

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None

# Default to OpenAI if available and key is set, else Groq
client = None
provider = "openai"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a NER schema with LLM.")
    parser.add_argument("--PATH_DATA", type=str, help="Path to the data files")
    parser.add_argument(
        "--text", type=str, help="name of the text file", default="example_death_act"
    )
    parser.add_argument(
        "--PATH_CONFIG", type=str, help="Path to the config file, schema and PROMPT", required=True
    )
    parser.add_argument(
        "--prompt", type=str, help="name of the prompt file", default="death_act"
    )
    parser.add_argument(
        "--schema", type=str, help="name of the schema file", default="death_act_schema"
    )
    parser.add_argument(
        "--provider", type=str, choices=["openai", "groq"], default=None, help="LLM provider to use"
    )
    
    args = parser.parse_args()

    # Determine provider
    if args.provider:
        provider = args.provider
    elif os.getenv("GROQ_API_KEY"):
        provider = "groq"
    else:
        provider = "openai"

    # Initialize client
    if provider == "openai":
        if not OpenAI:
            print("Error: OpenAI library not installed.")
            sys.exit(1)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment.")
            sys.exit(1)
            
        client = OpenAI(api_key=api_key)
        model = "gpt-3.5-turbo"
    elif provider == "groq":
        if not Groq:
            print("Error: Groq library not installed. Run 'pip install groq'")
            sys.exit(1)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("Error: GROQ_API_KEY not found in environment.")
            sys.exit(1)
            
        client = Groq(api_key=api_key)
        model = "llama-3.1-70b-versatile" # Tool calling supported model
    else:
        print("Error: Unknown provider")
        sys.exit(1)

    # Read Text
    file_path = join(args.PATH_DATA, f"{args.text}.txt")
    if not os.path.exists(file_path):
        # Fallback: maybe the arg IS the text content or a direct path
        if os.path.exists(args.text):
            file_path = args.text
        else:
             # Assume direct text if not a file? Or just fail. 
             # Existing script assumed it's in PATH_DATA with .txt extension.
             # Let's stick to existing logic but handle full path if provided.
             if os.path.exists(args.PATH_DATA) and os.path.isfile(args.PATH_DATA):
                 file_path = args.PATH_DATA
             else:
                 print(f"Error: Could not find text file at {file_path}")
                 sys.exit(1)
            
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
         print(f"Error reading text file: {e}")
         sys.exit(1)

    text = text.replace("(", "").replace(")", "")

    # Read Prompt
    prompt_path = join(args.PATH_CONFIG, f"{args.prompt}.prompt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()
    
    prompt = Template(prompt_content)
    content = prompt.render(text=text)

    # Read Schema
    schema_path = join(args.PATH_CONFIG, f"{args.schema}.json")
    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    # Call API
    print(f"Calling {provider} with model {model}...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            tools=[{"type": "function", "function": schema}],
            tool_choice="auto" if provider == "openai" else {"type": "function", "function": {"name": schema["name"]}},
            temperature=0,
            top_p=1.0,
        )

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            print(tool_calls[0].function.arguments)
        else:
            print("No entities found (no tool calls returned).")
            print("Response content:", response.choices[0].message.content)

    except Exception as e:
        print(f"Error during API call: {e}")
        sys.exit(1)
