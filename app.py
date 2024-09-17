import streamlit as st
import pandas as pd
from transformers import pipeline
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate

# Load the customer support ticket dataset
def load_data():
    # Adjust the path to your Excel file
    df = pd.read_csv("C:\\Users\\Anand\\Downloads\\archive\\customer_support_tickets.csv")
    return df

# Display dataset insights
def show_data(df):
    st.write("Dataset Preview:")
    st.dataframe(df)

# Sentiment analysis function using Hugging Face
def analyze_text(text):
    classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')
    result = classifier(text)
    return result

# LangChain analysis function
def analyze_conversation(conversation):
    template = """
    Analyze the following customer support conversation and provide insights:
    Conversation: {conversation}
    """
    prompt = PromptTemplate(template=template, input_variables=["conversation"])
    llm = OpenAI(model="gpt-3.5-turbo")  # Make sure you have your OpenAI API key set up
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run(conversation=conversation)

# Streamlit app UI
def main():
    st.title("Customer Support Conversation Analyzer")

    # Load dataset
    df = load_data()

    # Display dataset option
    if st.checkbox("Show Dataset"):
        show_data(df)

    # Process each row in the dataset
    if st.button("Analyze All Tickets"):
        for i, row in df.iterrows():
            st.write(f"**Ticket {i+1}:**")
            
            # Loop through each column in the row and display its content
            for col in df.columns:
                st.write(f"**{col}:** {row[col]}")

            # Example: Using a specific column for analysis (adjust based on your needs)
            if 'ticket_description' in df.columns:
                ticket_description = row['ticket_description']

                # Run Sentiment Analysis
                sentiment_result = analyze_text(ticket_description)
                st.write("Sentiment Analysis Output:")
                st.write(sentiment_result)

                # Run LangChain Analysis
                langchain_result = analyze_conversation(ticket_description)
                st.write("LangChain Analysis Output:")
                st.write(langchain_result)

if __name__ == "__main__":
    main()
