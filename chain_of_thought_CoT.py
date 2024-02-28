################################################################################
# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
# 
# Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, 
# Fei Xia, Ed Chi, Quoc Le, Denny Zhou
# 
# 28 Jan 2022
# 
# https://arxiv.org/abs/2201.11903
################################################################################



###################################
#  Set up environment
###################################

import os

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["FIREWORKS_API_KEY"] =""

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Chain of Thought CoT"


############################################################
#  Simple prompting
#  One-shot learning with a simple answer
############################################################

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.fireworks import ChatFireworks

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", 
         "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
         "Each can has 3 tennis balls. How many tennis balls does he have now?"),

        ("ai", 
         "The answer is 11."),

        ("human", 
         "The cafeteria had 23 apples. "
         "If they used 20 to make lunch and bought 6 more, how many apples do they have?")
    ]
)

cllm = ChatFireworks(model = "accounts/fireworks/models/mistral-7b-instruct-4k",)





############################################################
#  Chain-of-Thought prompting
#  One-shot learning with a CoT example answer
############################################################

prompt = ChatPromptTemplate.from_messages(
    [
        ('human', 
         "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. "
         "Each can has 3 tennis balls. How many tennis balls does he have now?"),

         ("ai",
          "Roger started with 5 balls. 2 cans of 3 tennis balls each is "
          "6 tennis balls. 5 + 6 = 11. The answer is 11."),
          
          ("human",
           "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, "
           "how many apples do they have?")
    ]
)



cot_chain = prompt | cllm

print("-"*10)
for chunk in cot_chain.stream({"": ""}):
    print(chunk.content, end="")
print("\n" +"-"*10)




pass