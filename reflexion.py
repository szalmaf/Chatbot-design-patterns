################################################################################
# Reflexion: Language Agents with Verbal Reinforcement Learning
# 
# Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, 
# Karthik Narasimhan, Shunyu Yao
# 
# Mar 2023
# 
# https://arxiv.org/abs/2303.11366
################################################################################



###################################
#  Set up environment
###################################

import os

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["TAVILY_API_KEY"] =""
# os.environ["FIREWORKS_API_KEY"] =""

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Basic Reflection"



###################################
#  Generate chain
###################################

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.fireworks import ChatFireworks

promptt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an essay assistant tasked with writing excellent 5-paragraph essays."
            " Generate the best possible essay for the user's request.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

cllm = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    model_kwargs={"max_tokens": 32768}
)

generate_chain = promptt | cllm


task = HumanMessage(content="Write an essay on why the Little Prince is relevant in modern childhood!")
# print("\n\n", "-"*20)
# essay = ""
# for chunk in generate_chain.stream(input={"messages": [task]}):
#     print(chunk.content, end="")
#     essay += chunk.content



###################################
#  Reflection chain
###################################

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a teacher grading student's essays. Generate critique and recommendations and"
         " add notes on the merits and failures of the essay. The detailed recommendations should include"
         " requests for length, depth and style."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

reflection_chain = reflection_prompt | cllm

# print("\n\n", "-"*20)
# reflection = ""
# for chunk in reflection_chain.stream(input={"messages": [task, HumanMessage(content=essay)]}):
#     print(chunk.content, end="")
#     reflection += chunk.content



###################################
#  Repeat generation using the given feedback
###################################

# print("\n\n", "-"*20)
# new_essay = ""
# for chunk in generate_chain.stream(input={
#     "messages": [
#         HumanMessage(content=task.content),
#         AIMessage(content=essay),
#         HumanMessage(content=reflection)
#         # Here the AI writes another, improved essay based on the above "conversation"
#     ]
# }):
#     print(chunk.content, end="")
#     new_essay += chunk.content



###################################
#  Simple reflection graph
###################################

from typing import Sequence, List
from langchain_core.messages import BaseMessage
from langgraph.graph import MessageGraph, END

def generator_node(messages: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": messages})

def reflection_node(messages: Sequence[BaseMessage]):
    switch_message_type = {"ai": HumanMessage, "human": AIMessage}
    reordered_messages = [messages[0]] + \
        [
            switch_message_type[message.type](content=message.content) \
            for message in messages[1:]
        ]
    reply = reflection_chain.invoke({"messages": reordered_messages})
    return HumanMessage(content=reply.content)

def should_continue(messages: List[BaseMessage]):
    if len(messages) > 6:
        return END
    return "reflect"


graph = MessageGraph()
graph.add_node("generate", generator_node)
graph.add_node("reflect", reflection_node)
graph.add_conditional_edges(
    start_key="generate",
    condition=should_continue
)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")
app = graph.compile()



###################################
#  Interact with graph
###################################

print("-"*10)
for event in app.stream([HumanMessage(content="Write an essay on why the Little Prince is relevant in modern childhood!")]):
    if list(event.keys())[0] == "__end__":
        print(list(event.keys())[0])
        print(list(event.values()))
        print("-"*10)
    else:
        print(list(event.keys())[0])
        print(list(event.values())[0].type)
        print(list(event.values())[0].content)
        print("-"*10)
print("-"*10)







pass