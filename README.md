# Research-Paper-Summarizer-For-Different-Personas-Using-LangChain-and-RAG
A project to summarize a research paper for specific personas with varying levels of subject knowledge utilizing LangChain and RAG


## Project Overview
This project provides a solution for summarizing research papers tailored to various professional and academic personas. It leverages advanced natural language processing (NLP) techniques, including embeddings, vector stores, large language models (LLMs) and a retrieval system, to generate concise and relevant summaries. Beyond summarization, the tool also includes an LLM-as-a-judge evaluation system to assess the quality of the generated summaries from each persona's perspective.


## Why It's Useful 
Exploring the extensive domain of research papers can be demanding and time-consuming. This application addresses this challenge by:
**1. Personalized Summaries**: Generating summaries specifically designed for different audiences (e.g., Data Scientists, AI Engineers, Business Executives, Graduate Students, General Audience), ensuring relevance and clarity for the intended reader.
**2. Time-Saving**: Quickly extracting relevant information from lengthy research papers, allowing users to grasp core concepts without reading the entire document.
**3. Enhanced Understanding**: Presenting information in a way that resonates with a user's background, making complex topics more accessible.
**4. Quality Assurance**: Including an evaluation mechanism to provide feedback on the summary quality, helping users refine their understanding or further explore specific aspects.
**5. Versatile Input**: Supporting various input sources, including local PDF files, ArXiv IDs, and ArXiv URLs.


## Technologies Used
The project utilizes the following Python libraries and frameworks:
**1. LangChain**: For orchestrating LLM interactions, document loading, text splitting, and prompt management.
**2. Anthropic API (Claude LLMs)**: For generating summaries and performing evaluations. Specifically, `claude-3-7-sonnet-20250219` for summarization and `claude-sonnet-4-20250514` for evaluation.
**3. HuggingFace Embeddings (`all-MiniLM-L6-v2`)**: For converting text into numerical representations (embeddings) to facilitate similarity searches.
**4. FAISS**: For similarity search and storage of document embeddings (vector store).
**5. NLTK**: For sentence tokenization during document processing.
**6. python-dotenv**: For securely managing API keys.
**7. logging**: For effective debugging and monitoring of the application's flow.


## Setup and Running the Project
To set up and run this project, follow this steps

### 1. Clone the Respository

```
git clone https://github.com/shrividya-subramaniam/research-paper-summarizer.git
cd research-paper-summarizer
```

### 2. Install Dependencies

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Obtain an Anthropic API Key
You will need an API key from Anthropic to use their Claude models. Visit the Anthropic website to sign up and obtain your key.

### 4. Configure Environment Variables
Create a file named `.env` in the root directory of your project and add your Anthropic API key:

```
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### 5. Run the Summarizer
The core logic is contained within the `research_paper_summarizer.ipynb` notebook. You can run it directly if you have Jupyter installed:

```
jupyter notebook research_paper_summarizer.ipynb
```

#### Supported Input Sources
1. Local PDF file: Ensure the PDF is in the same directory or provide the full path)
2. ArXiv paper by ID
3. ArXiv URL


## Examples of Input and Output

### Input
This project demonstrates summarization using "Attention is All You Need" as a sample research paper. The publicly available PDF was used as input for the summarization process. 

**Citation:**
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. (2017). *Attention Is All You Need*. arXiv. Retrieved from [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

```
# Example for a local PDF file
source = "Attention is All You Need.pdf"

# Example for an ArXiv paper using its ID
# source = "arxiv:1706.03762" # ArXiv ID for Attention is All You Need Paper

# Example for an ArXiv paper using its URL
# source = "https://arxiv.org/abs/1706.03762" # URL for Attention is All You Need Paper

# Custom query for focused summarization
query = "Summarize this research paper" 
```

### Output
The output will be a structured set of summaries, one for each defined persona, along with an evaluation of each summary. Due to the length, only a snippet is shown.

```
================================================================================
PAPER SUMMARIZATION RESULTS
================================================================================

Source: Attention is All You Need.pdf
Query: Summarize this research paper
LLM Model Used: claude-3-7-sonnet-20250219
Processing: Processed 373 sentence windows
Context Length: 61,788 characters

------------------------------------------------------------
PERSONA SUMMARIES
------------------------------------------------------------

--- DATA SCIENTIST ---
# Comprehensive Summary of "Attention Is All You Need"
## 1. Main Problem/Research Question
The paper addresses limitations in sequence transduction models that rely on complex recurrent or convolutional neural networks with encoder-decoder architectures. ...

--- AI ENGINEER ---
# Engineering Analysis: "Attention Is All You Need" Paper
## 1. Main Problem/Research Question
The paper addresses limitations in sequence transduction models that rely on complex recurrent or convolutional neural networks with encoder-decoder architectures. ...

... (Summaries for Graduate Student, Business Executive, General Audience) ...

------------------------------------------------------------
SUMMARY EVALUATIONS
------------------------------------------------------------

--- DATA SCIENTIST EVALUATION ---
**Rating: 4 (Very Good)**

## Detailed Justification:

### Accuracy (4.5/5)
The summary accurately represents the key aspects of the Transformer paper. ...

--- AI ENGINEER EVALUATION ---
**Rating: 4 - Very Good**

## Detailed Justification:

### Accuracy (4.5/5)
The summary accurately represents the source material with high fidelity. ...

... (Evaluations for Graduate Student, Business Executive, General Audience) ...
```


## Challenges Faced and Solutions
During the development and testing of this project, two challenges were encountered.

### 1. Lack of Native Embedding Models from Anthropic

**Problem:** 
Unlike OpenAI, Anthropic does not offer proprietary embedding models. This meant a separate solution was required to generate embeddings. 

**Solution:**:
To address this, an open-source embedding model from HuggingFace, specifically all-MiniLM-L6-v2 model, was integrated into the project. This model is lightweight, efficient, and performs well for semantic similarity tasks.The generated embeddings are then stored and managed using FAISS (Facebook AI Similarity Search), an open-source library for similarity search and clustering of dense vectors.This combination allowed the project to create a knowledge base from the research papers, enabling the sentence window retrieval mechanism to work without depending on Anthropic for embeddings.

### 2. API Rate Limits and Model Overloading (Anthropic API)
**Problem:**
Initially, when making multiple consecutive calls to the Anthropic API, particularly for both summarization and evaluation for several personas, the API was returning a `429 Too Many Requests` error. This indicated that the request rate was exceeding the allowed rate limit (40k ITPM per minute for Anthropic Claude API), leading to incomplete summary and evaluation results. 

**Solution:**
To resolve this issue, a strategic time.sleep(30) delay was introduced after each LLM call (both for generating summaries and for evaluating them). This pause helps to:

1. Respect API Rate Limits: By introducing a delay, the frequency of requests to the LLM API is reduced, making it less likely to hit the predefined rate limits.
2. Prevent Overloading: Giving the API server a brief break between requests allows it to process previous queries and reduces the chance of receiving an `Too Many Requests` error.

This simple solution significantly improved the success rate of the summarization and evaluation processes.



