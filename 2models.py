# Inspired by https://github.com/mneedham/LearnDataWithMark/blob/main/ollama-parallel/app_v2.py
# Previously, two models were the maximum; now, two models are the minimum.

"""
Here are the main differences:

    Initialization:
        The first code uses ollama.AsyncClient() to initialize the client.
        The second code uses AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ignore-me") to initialize the client.

    Session State Initialization:
        The first code initializes session state variables in a loop for SESSION_KEYS.
        The second code directly initializes messages1, messages2, and input_disabled.

    Model Selection:
        The first code dynamically selects the number of models using a number input and loops through model selection.
        The second code statically selects two models (model_1 and model_2).

    Prompt Handling:
        Both handle prompt input and submission, but the first code uses st.chat_input with an on_submit method.
        The second code uses st.chat_input with an on_submit method but has additional GIF loading for responses.

    Async Function Differences:
        The first code's run_prompt function calls await client.chat(model=model, messages=messages, stream=False).
        The second code's run_prompt function streams the response and updates the UI incrementally with await client.chat.completions.create(model=model, messages=messages, stream=True).

    Styling:
        The first code has custom CSS included directly in the script.
        The second code has similar CSS but also includes a GIF for loading animation.

    Clearing State:
        The first code has clear_session_state() to clear messages and reset input.
        The second code has clear_everything() to clear messages and reset input.

function llmstart {
    $env:OLLAMA_ORIGINS="app://obsidian.md*"
    $env:OLLAMA_DEBUG="1"
    $env:OLLAMA_HOST="0.0.0.0"
    $env:OLLAMA_NUM_PARALLEL="4" 
    $env:OLLAMA_MAX_LOADED_MODELS="4" 
    ollama serve
}
streamlit run .\2models.py

"""
import ollama
import streamlit as st
import asyncio
from streamlit_extras.stylable_container import stylable_container

# Set up the page
PAGE_TITLE = "Running LLMs in parallel with Ollama"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# Custom CSS for styling
CUSTOM_CSS = """
<style>
hr {
    margin: -0.5em 0 0 0;
    background-color: red;
}
p.prompt {
    margin: 0;
    font-size: 14px;
}

div.stChatMessage :has(div[data-testid="chatAvatarIcon-assistant"]) {
    flex-direction: row-reverse;
    text-align: right;
}

img.spinner {
    margin: -1em 0 0 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state variables
SESSION_KEYS = ["input_disabled"]
for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = False

# Initialize client and models
client = ollama.AsyncClient()
models = [model['name'] for model in ollama.list()["models"] if model["details"]["family"] in ["llama", "gemma"]]

# Functions to clear and disable input
def clear_session_state():
    for key in list(st.session_state.keys()):
        if key.startswith("messages"):
            st.session_state[key] = []
    st.session_state["input_disabled"] = False

def disable_input():
    st.session_state["input_disabled"] = True

# Sidebar for model selection and prompt input
with st.sidebar:
    with stylable_container("blue", css_styles="button { background-color: red; color: white; }"):
        st.button("New Chat :speech_balloon:", on_click=clear_session_state)
    st.write("***")

    num_models = st.number_input("Number of Models", min_value=2, value=2, step=1)
    selected_models = []
    for i in range(num_models):
        model = st.selectbox(f"Model {i+1}", options=models, index=models.index("phi3:latest") if i == 0 else models.index("llama3:latest"), disabled=st.session_state["input_disabled"])
        selected_models.append(model)
        if f"messages{i+1}" not in st.session_state:
            st.session_state[f"messages{i+1}"] = []

    st.markdown("<p class='prompt'>Prompt</p>", unsafe_allow_html=True)
    prompt = st.chat_input("Message Ollama", on_submit=disable_input)

# Layout for displaying chat messages
placeholders = []
columns = st.columns(num_models)
for i, col in enumerate(columns):
    col.write(f"# :blue[{selected_models[i]}]" if i % 2 == 0 else f"# :red[{selected_models[i]}]")
    placeholders.append(col.empty())

# Function to run the prompt and display messages
async def run_prompt(placeholder, model, message_history):
    with placeholder.container():
        for message in message_history:
            st.chat_message(name=message['role']).write(message['content'])
        assistant = st.chat_message(name="assistant")

    messages = [{"role": "system", "content": "You are a helpful assistant."}, *message_history]
    response = await client.chat(model=model, messages=messages, stream=False)
    
    assistant_message = response['message']['content']
    message_history.append({"role": "assistant", "content": assistant_message})
    
    with placeholder.container():
        for message in message_history:
            st.chat_message(name=message['role']).write(message['content'])
        assistant.write(assistant_message)

# Main function to run prompts in parallel
async def main():
    tasks = []
    for i, model in enumerate(selected_models):
        tasks.append(asyncio.create_task(run_prompt(placeholders[i], model=model, message_history=st.session_state[f"messages{i+1}"])))
    await asyncio.gather(*tasks)
    st.session_state["input_disabled"] = False

# Handle prompt submission
if prompt:
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        for i in range(num_models):
            st.session_state[f"messages{i+1}"].append({"role": "user", "content": prompt})
        asyncio.run(main())