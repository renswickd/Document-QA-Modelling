from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import CTransformers
# from dotenv import find_dotenv, load_dotenv

# # Load environment variables from .env file
# load_dotenv(find_dotenv())


qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'})

vectordb = FAISS.load_local(r"./vectorstore/", embeddings)

llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 256,
                            'temperature': 0.1}
                    )

qa_prompt = PromptTemplate(template=qa_template,
                        input_variables=['context', 'question'])

dbqa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vectordb.as_retriever(
                                        search_kwargs={'k': 3}),
                                    chain_type_kwargs={'prompt': qa_prompt}
                                    )


