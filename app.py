import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
from dotenv import load_dotenv
import traceback
### Load environment variables
load_dotenv()

# Streamlit App Config
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.markdown("**Easily generate summaries of YouTube videos or web content.**")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Enter a valid YouTube video URL or website URL.
2. Click "Summarize the Content."
3. Wait for the summary to appear!
""")

# Input fields
groq_api_key = os.getenv("GROQ_API_KEY")
generic_url = st.text_input("Enter URL (YouTube or Website):", label_visibility="visible")

# LLM setup
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button click event
if st.button("Summarize the Content"):
    # Input validation
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the required information to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It must be a YouTube video or a website link.")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                                          "Chrome/116.0.0.0 Safari/537.36 Edg/114.0.0.0"
                        },
                    )
                docs = loader.load()

                # Summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display summary
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")
