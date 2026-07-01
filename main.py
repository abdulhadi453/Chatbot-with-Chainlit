from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv()
set_tracing_disabled(disabled=True)

openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")

external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

model = OpenAIChatCompletionsModel(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_client=external_client
)


config = RunConfig(
    model=model,
    model_provider=external_client,
)

agent = Agent(
    name = "General Agent",
    instructions = "You are general purpose agent that can help with any question and user query. Whenever asked you who build and developed you, you should answer that you are developed by Full Stack Developer named Abdul Hadi.",
    model=model
)

@cl.on_chat_start
async def handle_chat():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello Greetings! How may I assist you today?").send()
    
@cl.on_message
async def on_message(message : cl.Message):
    msg = cl.Message(content='')
    await msg.send()
    
    history = cl.user_session.get("history")
    history.append({"role":"user", "content": message.content})

    result = Runner.run_streamed(
        agent,
        input=history
    )
    
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    history.append({"role":"assistant", "content": result.final_output})
    
    cl.user_session.set("history",history)