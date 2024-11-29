---
License: apache-2.0
Title: Youtube Video and Website Content summary generator
Sdk: streamlit
Emoji: 💻
ColorFrom: red
ColorTo: yellow
Short_description: Summarizer using LLM
---


📋 Project Overview
    This Streamlit application provides an easy way to summarize content from YouTube videos or websites using LangChain and Groq. The app leverages LLM 
    capabilities to extract and condense key information, enabling users to quickly grasp the essence of the content in 300 words.

🛠️ Features
    Content Summarization: Summarize YouTube videos or web pages into concise summaries.
    User-Friendly Interface: Intuitive Streamlit UI for smooth user interaction.
    Automated Content Fetching: Dynamically loads and processes data from the provided URLs.
    Customizable Prompt: Uses a flexible prompt template to generate summaries.


🧑‍💻 Technology Stack
    Streamlit: For the web interface.
    LangChain: To handle the summarization workflow.
    Groq: Large language model for generating summaries.
    Python Libraries:
    validators: To validate URL inputs.
    langchain_groq: For connecting LangChain with Groq's API.
    dotenv: To securely load environment variables.
    langchain_community.document_loaders: To fetch content from YouTube and websites.
