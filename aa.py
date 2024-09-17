import pandas as pd

# Load dataset
df = pd.read_csv("C:\Users\Anand\Downloads\archive\customer_support_tickets.csv")
print(df.head())


from transformers import pipeline

# Load small text classification model
classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')

# Test the model
result = classifier("The product I received is damaged and I want a refund.")
print(result)

from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

template = """
Analyze the following customer support conversation and provide insights:
Conversation: {conversation}
"""

# Create a prompt
prompt = PromptTemplate(template=template, input_variables=["conversation"])

# Using OpenAI API (you may need your OpenAI API key)
llm = OpenAI(model="gpt-3.5-turbo")

chain = LLMChain(prompt=prompt, llm=llm)

# Test conversation
conversation = """
Customer: I am facing an issue with my order. It hasn’t arrived yet.
Support: I apologize for the delay. Let me check the status of your order.
"""

# Run LangChain agent
output = chain.run(conversation=conversation)
print(output)
