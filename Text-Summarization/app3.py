import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize PDF, YT, or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize PDF, YT, or Website")
st.subheader('Summarize Content')

## Get the Groq API Key and URL or PDF file to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter URL (YT or Website)", label_visibility="collapsed")
uploaded_pdf = st.file_uploader("Or upload a PDF", type=["pdf"])

## Groq Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize Content from URL or PDF"):
    ## Validate all inputs
    if not groq_api_key.strip() or (not generic_url.strip() and not uploaded_pdf):
        st.error("Please provide the information to get started (URL or PDF)")
    elif generic_url and not validators.url(generic_url):
        st.error("Please enter a valid URL or upload a PDF")
    else:
        try:
            with st.spinner("Processing..."):
                docs = []
                
                ## If YouTube or Website URL
                if generic_url:
                    if "youtube.com" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                ## If PDF is uploaded
                elif uploaded_pdf:
                    # Convert the uploaded PDF file to a file-like object
                    pdf_bytes = BytesIO(uploaded_pdf.read())
                    loader = PyPDFLoader(pdf_bytes)
                    docs = loader.load()

                ## Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
