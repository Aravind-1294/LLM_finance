import streamlit as st
import os
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone,Chroma
from langchain.chains import ConversationalRetrievalChain
import pathlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import Cohere, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import faiss
from openai import OpenAI
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.embeddings import CohereEmbeddings
import textwrap
from googlesearch import search
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
from collections import Counter
import numpy as np
from langchain_openai import ChatOpenAI
import json
from langchain_community.retrievers import (
    QdrantSparseVectorRetriever,
)
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import MergerRetriever
from langchain_community.vectorstores import Qdrant
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import os
from langchain.agents import AgentType, initialize_agent,load_tools
import google.generativeai as genai



def sparse_encoder(chunk:str):
    with open('/home/aravind/Desktop/Major_project/vocab.json', 'r') as f:
      vocab = json.load(f)
    stop_words = set(stopwords.words('english'))
    clean_chunk = re.sub(r'[^a-zA-Z\s]', ' ', chunk)
    tokens = word_tokenize(clean_chunk)
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    token_counts = Counter(tokens)
    indices = []
    values = []
    for token, count in token_counts.items():
        if token in vocab:
            indices.append(vocab[token])
            values.append(count)
    return tuple([indices, values])


def get_bot_response(user_input):
    prompt2 =("""you are a Financial chatbot called wizbot,use the Instruction provided in quotations "
Life Cycle : When we do a Life Cycle Analysis of an industry, it is important to know which life cycle stage that industry is in. Every industry has 4 stages in its life cycle. First is the Pioneering Stage, then the growth stage, then the saturation or moderation stage and finally the declining stage. In this declining stage, companies are not transforming but are declining. In these 4 stages, the pioneering stage is when companies are in the starting stage. If we can pick up the companies in this stage, there will be more risk but also more reward. If we see the growth stage from the pioneering stage, the companies in this stage are established and have seen phenomenal growth. If we talk about India, we can say that the internet industry is in the growth stage. It is growing in double digit and higher in double digit. So, if we pick up companies in this stage or in this industry, we will see very rapid growth. Again, there is a risk in this stage also. Next is maturity or saturation stage. In maturity or saturation stage, In this maturity or saturation stage, if the companies grow rapidly for 10-15 years, the industry reaches a peak cycle. When it reaches this peak cycle, the companies focus on volume growth and focus on premiumization. And the last stage is the declining stage. In this declining stage, the industries are actually revolving.

examples for Life Cycle of a company : To tell you practically, In India, electric vehicles are in A-stage. Or green energy industry is in A-stage. This is called pioneering stage. Basically, companies are still thinking about setting up a business in the industry. Sometimes, they are setting up. When some companies enter, this is called pioneering or starting stage.
if we see the soaps industry, who are the people who use soaps? Everyone uses soaps industry, who are the people who use soaps? Everyone uses soaps right? Soaps industry has reached the saturation or maturity stage from the growth stage to the pioneering stage. Similarly, if we talk about the two wheeler industry, almost everyone has a two wheeler at home right? So, if we look at the two wheeler industry 10 years back, it was in the growth stage. But now it is in the maturity or saturation stage. What companies do in this stage is, as volume growth does not happen, they focus on premiumization. For example, they sell special segment vehicles or special segment products. So that they use the brand and sell it at a higher price and earn more revenue. This is the certain stage. Basically, if we invest in any company in this revenue by selling at a lower rate. This is the second stage. Basically, if we invest in any company in this stage, it is a little safe. Because, they have already grown to a certain level, so their base has been formed. So, we have less risk, but also less reward.  For example, if we talk about it now, see, as you remember, steam engines used to come in trains. Right? In the new generation of steam engines, they used to grow from the pioneering stage, matured, then steam engines, diesel engines, electric engines, they used to come. So, every industry is looking at this cycle.
When we are in the declining stage, then definitely it is better to avoid such companies. Because, even though these companies are good companies after declining stage, there will be no big movement in the stock price. And we might see degrowth in the stock price too from these companies. So, when we invest in a company, it is better to know which industry it is in and which life cycle it is in. Because if we invest in the first two life cycle stages, we might make very big money. But if we invest in a company after the third stage, we will lose money. So, we should know which life cycle stage the industry is in and invest in any company, we will lose money. So, we should know the life cycle of the industry and invest. So, how do we know? For example, some industries have a short life cycle. Like, if we talk about the PPE kits manufactured after COVID, many companies started investing in them. How long is the life cycle of the companies manufacturing PPE kits? The life cycle of the companies that manufacture the vaccine is short-lived life cycle. When we invest in any company, there should be scope for the company or industry to grow. For example, if we talk about internet-based companies, how many people are there in India, how many people are using the internet in that population, so that the scope of internet growth and penetration, if we understand this, the more the internet grows, the companies we invest in, if the company is a good company, it will grab maximum market share. So that the company will grow along with the industry.

Quantitative analysis of an industry : After doing a life cycle analysis of an industry, if we want to do a qualitative and quantitative analysis of an industry, Michael Porter's 5 forces analysis will be helpful for us. use Michael porter's 5 forces analysis.

Michael porters 1st Force : he says that we should see the level of competition in the industry. If the competition is high, then companies cannot earn more margins. Because many companies offer products at low prices. When they offer, the profitability of the company does not grow. Volume will grow, but profitability will not grow. Do you remember any industry here? I will tell you the industry I remember.
examples : Stock Broking Industry If we see the past few months, the volume of the stock broking industry is growing. But due to the increase in competition like zerodha, angel broking By entering all such discount brokers Actually, what happens is that the profitability has reduced a lot. Because of that, many companies are actually struggling. So, it is better not to invest in such industries Many companies are struggling because of this. So, it is better not to invest in such competitive industries. Because, the company's growth may not be big. Next,

 Michael porters 2nd Force : Threat of New Entrants. Basically, if someone is a new entrant, if he is a strong entrant, he might disrupt the entire industry.
examples : we are talking about the telecom industry. When did Jio enter the telecom industry. When Jio entered the telecom industry, when Mukesh Ambani launched Jio, to grab the market share, at a very low price, the competition offered the prices, especially in the first, they gave it for free. After that, the price offered was also very low. This disrupted entire industry. Because of that, the industry is under a lot of pressure on profitability. So, when we invest in any industry, we should analyze if there is a strong new entrant coming in. I see this kind of threat in the future in the retail segment. As we have seen, the profitability in the retail segment is very high. Only avenue supermarket is earning profitability. But, now, Jio, now they are getting jio mart with facebook so they might disrupt this market in future .

Michael porters 3rd force : threat of substitute if the product is substituted to the product of the, then again it is a threat to the industry.
examples : If we want to talk practically, if we want to watch TV, we watch through a dish TV or a wired TV. But, now a days, there is a very aggressive substitute for this, like OTT platforms. Netflix, Amazon, Hotstar, all these are coming. So, if people get used to them, if they reduce watching TV, then, it is a threat to TV. The threat to camera is the threat to mobile. Even if there is a substitute, it is a threat to the industry. We should be aware of this too.

Michael porters  4th force : bargaining power of the supplier. if the supplier has more bargaining power than the company, then our investment company will not make much money.
examples : To give you an example, if we talk about the passenger vehicle segment in India, the major market share is with Maruti. Various companies supply parts parts to Morthy. So, the companies that supply the parts, when they buy the product from those companies, they will bargain it to the maximum extent. Because if this company supplies, Maruti can buy it from some other company. Because if there is no uniqueness in the product, this company can bargain it or this company can demand that margins. But if there is no uniqueness in the product, if other companies can offer the same product, what Maruti does is, he tries to grab maximum margins. So, I am giving examples of Maruti here. But, this happens in many industries. So, if we invest in the same company, if the company has a limited client base, then it is a risk. There are very few clients who have client less, they will grab maximum margins. So, we should remember this threat.

 Michael porters  5th force : bargaining power of clients. If the buyers have bargaining power, if the buyers are not buying the products, the companies will offer them at a low price. If they offer like that, the company will not make much money.
So, when we analyze any industry, we should remember these 5 forces of Michael Porter's company. We should check the threats of the company in which we invest in these 5 forces. Then, investing would be a better option.

Additional points to check : One is high entry barrier. If the company or the industry has a high or industry, then it is a good thing. Because the chances of new competition are very less and their margins are also more.
examples : Like we want to talk about IRCTC has a high entry barrier. Because it is very difficult for the competition to come in the ticketing business. We also interviewed Mr. Sourav Mukherjee. He also explained about Nestle. He said that Nestle has a high entry barrier in the baby milk powder segment. There are various other companies that have high entry barriers. If we invest in such companies, we might make very big money. Over a long period, we will make very big money. So, we should check if the industry has a high entry barrier.

Goverment protection : Also, we have to check if the government policies support it. After this high entry barrier, we have to see another point, which is government protection. Many times, the government offers protection to various segments.
examples : Like banking segment. If it comes to banking segment, whoever falls, they will not get a banking license. So, if I want to open a bank in the morning, it will not happen. They have to take permission from the RBI, after taking permission, they have to set up various rules and regulations. Similarly, for example, we have to talk about the Royal Enfield. Okay, in the motorbike segment, we have seen that Harley Davidson has come to India, but to sell Harley Davidson's bikes, basically the bike will have a cost, actually, almost equivalent equivalent to taxes. Because of this, Royal Enfield sells their bikes at a higher price than the price of the bike. So, buyers do not get caught. In this way, the government protects the industries by buying. If we invest in such a protected industry, it will be very helpful for the companies.
So, if you invest in a company, company is in a high fragmented market, that is, if the market is in a market with a lot of players, then it is better to avoid such companies.
Also, if you can sell a commodity, for example, a product that anyone can make, and they do not need any rules, they do not need government permissions, that is, if they need small permissions, they can easily set up production and sell it. It is better to avoid such companies.

Top 10 Companies by Market Share:

Because those companies will not earn much revenue growth and will not create good wealth for us. So, as we have seen, when we do an industry analysis, the key main 3 points to remember while doing analysis are 1. See the scope of the industry growth 2. See the scope of the industry profitability growth 3. Check the risks of the industry If we keep these three points in mind and analyze them through the parameters we discussed earlier We can see the scope of the industry grow and profit margins are also there, if the threats are less, then we should try to buy maximum shares of such companies."  based on the instruction and examples provided, give me the complete sector analysis of {sector} industry in india explain each point in detail required for making an investment in detail.""")
    run2 = PromptTemplate(input_variables=["sector"], template=prompt2)
    super1 = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key='AIzaSyA10AMAbb1AoTs4rHdHBCG7oDDZWY-VgN4')
    chain2 = LLMChain(
    llm=super1,
    prompt=run2,
    verbose=True,
    )
    Analyze = chain2.predict(sector = user_input)
    return Analyze


def get_embedding(text, model="text-embedding-ada-002"):
   client = OpenAI(api_key="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")
   text = text.replace("\n", " ")
   return np.array(client.embeddings.create(input = [text], model=model).data[0].embedding)




def main():
    st.set_page_config(layout="wide", page_title="RatioWiz")
    st.sidebar.title("Features")
    page = st.sidebar.radio("Go to", ["Ratio WIZ Chatting","Sector Analysis","Sector chatbot","News Sentiment","Annual Reports"])


    if page == "Ratio WIZ Chatting":
        st.title("Ratio WIZ Chatting")
        st.write("Get the 1 month chat of your own Company/Ticker 📈📉")
        company_name = st.text_input("Company Ticker:", "")
        msft = yf.Ticker(company_name)
        hist = msft.history(period="1mo")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"]) )
        st.plotly_chart(fig)
        


    if page == "Sector Analysis":
        st.title("Sector Analysis 📈📉")
        st.markdown("""
    Sector Analysis ! 🚀 Where you can analyse and understand about the sector Of Indian Stock Market.
    """)
        user_input = st.text_input("Sector Name:", "")
        if st.button("Analyse"):
            with st.spinner('Analyzing Data ...'):
                if user_input:
                    bot_response = get_bot_response(user_input)
                    st.write(bot_response)
                else:
                    st.warning("Please type a message")



    if page == "Sector chatbot":
        st.title("Sector Analysis 📈📉")
        embed = CohereEmbeddings(cohere_api_key='8O4LKHnPlGXOf16TItwxm5WaK8b87k1jZ6APNv43')
        db = FAISS.load_local('Sector_data',embed,allow_dangerous_deserialization=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if input := st.chat_input("What is up?"):
            with st.chat_message("user"):
                st.markdown(input)
            st.session_state.messages.append({"role": "user", "content": input})
            query = input
            context = db.similarity_search(query)
            prompt1 =("""you are a Financial chatbot called wizbot,your main job is to answer the query only from the provide overview of the provided text,text : {text}
                      human : HI
                      Wizbot : Hi how can i help?
                      human : who are you/
                      Wibot : i am Wizbot a sector analysis bot
                      human : {query}""")

            run1 = PromptTemplate(input_variables=["text","query"], template=prompt1)
            super = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key='AIzaSyA10AMAbb1AoTs4rHdHBCG7oDDZWY-VgN4')

            chain1 = LLMChain(
                    llm=super,
                    prompt=run1,
                    verbose=True,
                    )
            overview = chain1.predict(text = context,query = query)
            response = f"Echo: {overview}"
            with st.chat_message("assistant"):
                 st.markdown(response)
                 st.session_state.messages.append({"role": "assistant", "content": response})



    if page == "News Sentiment":
        llm = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key='AIzaSyA10AMAbb1AoTs4rHdHBCG7oDDZWY-VgN4')
        tools = [YahooFinanceNewsTool()]
        agent_chain = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,)
        tools = load_tools(["google-scholar", "google-finance"], llm=llm,serp_api_key ="5fac0c3fecdbb49f5712a370a8be9a0ff96a4af0b15d12e2af4b7bf015731348" )
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        st.title("News Sentiment Analysis 📜")
        st.markdown("""
    News sentiment! 🗞️ Where you can get classified News by company and their sentiment 
    """)
        user_input = st.text_input("Company Name:", "")
        if st.button("Get news"):
            with st.spinner('Getting news ...'):
                if user_input:
                    prompt1 =(""" You are a News chatbot, Your main task is to provide the latest or recent within last 24 hrs  news of the {company} also segregate the sentiment of the
                              overall news into 3 classes [positive,negative,neutral] along with the percentage of the sentiment, also provide a valid reason.
                              if people ask any other query other that stock maket company or sector tell them to provide a valid input""")
                    run1 = PromptTemplate(input_variables=["company"], template=prompt1)
                    super = ChatGoogleGenerativeAI(model='gemini-pro',google_api_key='AIzaSyA10AMAbb1AoTs4rHdHBCG7oDDZWY-VgN4')
                    chain1 = LLMChain(
    llm=super,
    prompt=run1,
    verbose=True,
    )
                    sentiment = chain1.predict(company = user_input)
                    st.write(sentiment)
                else:
                    st.warning("Please type a message")
    


    if page=='Annual Reports':
        st.title("Annual Reports")
        st.markdown("""
    Talk ! to the Annual Reports and Save time
    """)
        cohere_client = cohere.Client("TBVEwMdI2gEhGDrSClPYR5P8MZLEaBNJl34EmiRW")
        qdrant_client = QdrantClient(url="https://68c3bd7c-85e8-4ddd-8da6-6d889578e82b.us-east4-0.gcp.cloud.qdrant.io:6333", api_key="G3knmZ-UwEnAyi27lvmVruwAos2I86KfnLIhu7qdLy0p8iqdGMvg-g",)
        with open('/home/aravind/Desktop/Major_project/vocab.json', 'r') as f:
            vocab = json.load(f)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if input := st.chat_input("What is up?"):
            with st.chat_message("user"):
                st.markdown(input)
            st.session_state.messages.append({"role": "user", "content": input})
            with st.spinner('Getting Data ...'):
                llm = ChatOpenAI(openai_api_key = '',model_name='gpt-3.5-turbo',temperature=0)
                memory = ConversationBufferWindowMemory(memory_key="chat_history",k=5,return_messages=True)
                sparse_retriever = QdrantSparseVectorRetriever( client=qdrant_client,
                                collection_name="VectorEmbeddings",
                                sparse_vector_name="text-sparse",
                                sparse_encoder=sparse_encoder,
                                content_payload_key = "chunk",
                                metadata_payload_key = "payload"

                            )
                embeddings = CohereEmbeddings(cohere_api_key = 'TBVEwMdI2gEhGDrSClPYR5P8MZLEaBNJl34EmiRW', model="embed-english-v3.0")
                qdrant = Qdrant(qdrant_client,"VectorEmbeddings", embeddings, vector_name = "text-dense",content_payload_key = "chunk",metadata_payload_key = "payload")
                dense_retriever = qdrant.as_retriever(search_type = "similarity")
                lotr = MergerRetriever(retrievers=[dense_retriever, sparse_retriever])

                qa = ConversationalRetrievalChain.from_llm(llm,retriever=lotr,memory=memory)
                question = input
                result = qa({"question": question})
                results = result['answer']
                response = f"Echo: {results}"
                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()