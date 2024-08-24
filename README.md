
# KnowledgeBridgeLLM

This guide outlines the steps required to set up and run the KnowledgeBridgeLLM project. Follow the instructions carefully to ensure a smooth setup process.

## Project Overview
KnowledgeBridgeLLM is a project developed as part of the thesis "AUGMENTING INTELLIGENCE: BRIDGING LLMS WITH EXTERNAL KNOWLEDGE REPOSITORIES." The project focuses on integrating Large Language Models (LLMs) with both structured and unstructured financial data sources to enhance decision-making and data analysis. This repository demonstrates the application of vector-based retrieval (using ChromaDB) and knowledge graph-based retrieval (using Neo4j) for complex financial queries, showcasing how these approaches can be combined to leverage the strengths of each.

The primary goal is to evaluate and compare these retrieval methodologies to determine the most effective approach for handling financial data, while providing setup instructions, detailed code, and necessary resources for replicating the project.

## Project Components Explanation

The `KnowledgeBridgeLLM` project is structured into several core components, each fulfilling specific roles within the data processing, query handling, and model evaluation pipeline. Here is a detailed explanation of each component:

### `knowledge_graph_loader.py`
- **Purpose**: Manages the setup and creation of the knowledge graph for Approach 2.
- **Key Responsibilities**:
  - Handles the entire pipeline from data preprocessing to graph generation.
  - Integrates with Neo4j to build a graph database using nodes and relationships extracted from financial data.
  - Ensures that the graph structure is optimized for complex financial queries.

### `knowledge_graph_query.py`
- **Purpose**: Manages the query operations for the knowledge graph approach.
- **Key Responsibilities**:
  - Executes Cypher queries against the Neo4j graph database.
  - Integrates with the LLM for hybrid retrieval by combining structured graph data and unstructured vector data.
  - Provides responses based on the results retrieved from both the graph and vector databases.

### `EDA.ipynb`
- **Purpose**: Contains all Exploratory Data Analysis (EDA) tasks related to the project.
- **Key Responsibilities**:
  - Performs detailed analysis and visualizations on the financial datasets.
  - Helps identify key patterns, trends, and relationships in the data, aiding in the preprocessing steps.

### `data_preprocessing.py`: DataPreprocessor Class
- **Purpose**: Manages initial data loading and formatting.
- **Key Responsibilities**:
  - Loads data directly from Hugging Face in the initialization (`__init__`) function.
  - Processes data to ensure it is in the correct format for analysis and integration.

### `vector_index_loader.py`: IndexBuilder Class
- **Purpose**: Central to data indexing and embedding generation.
- **Key Responsibilities**:
  - Invokes `DataPreprocessor` for initial data preprocessing.
  - Converts preprocessed data into a DataFrame for easier manipulation.
  - Splits data into chunks for manageable processing and analysis.
  - Generates data embeddings and manages the embedding database.
  - Checks for existing embeddings in the vector database (chroma db) to either load them or create and save new embeddings as needed.

### `query.py`: QueryEngine Class
- **Purpose**: Handles processing and responding to user queries.
- **Key Responsibilities**:
  - Utilizes `IndexBuilder` for data indexing and access.
  - Loads embeddings from the "financial_reports" collection in the vector database.
  - Processes user queries by converting them to embeddings and fetching similar documents.
  - Creates prompts for the LLM to summarize fetched documents, then sends summaries back to `chat_ui.py` for display.

### `chat_ui.py`
- **Purpose**: Provides the user interface for the application.
- **Key Responsibilities**:
  - Acts as the entry point, calling `QueryEngine` to process user queries.
  - Uses Streamlit to create an interactive chat interface for input and display.
  - Passes user queries to `QueryEngine` and displays responses, maintaining a query history.

### `evaluation.py`
- **Purpose**: Independently evaluates the model's performance.
- **Key Responsibilities**:
  - Operates separately from the main application, providing a means to test and validate the model.
  - Loads and processes data, then selects chunks to create questions and answers.
  - Uses the model to generate responses to the questions and compares these to the LLM-generated answers based on the data chunks.
  - Produces results from the comparison, offering insights into model performance.

**Visual Guides**:
- For a comprehensive understanding of the interaction between these components, refer to the "RAG Architecture" flowchart in the flow diagram folder.
- The evaluation process and its workflow can be better understood by examining "Evaluation-Flow.png," also located in the flow diagram folder.

**TestOutcome**:
- The repository includes a `TestOutcome.csv` file, which documents the outcomes of tests previously conducted. This file serves as a benchmark for evaluating the model's performance against established metrics.

**Note**: Running the code requires an "OPENAI_API_KEY." Please ensure you have access to this key for testing purposes. If needed, contact the repository owner for access.


## Prerequisites

Before you begin, make sure you have Conda installed on your system to manage environments and dependencies.

## Setup Instructions

### Step 1: Environment Setup

First, open your terminal and navigate to the root directory of your project. If you're not already there, use the `cd` command to move to the correct directory.

1. **Deactivate any existing Conda environments to start fresh:**
   ```
   conda deactivate
   ```
   
2. **Create a new Conda environment:**
   Replace `your_env_name` with a name of your choice for the environment.
   ```
   conda create -n your_env_name python=3.10.6
   ```
   
3. **Activate the newly created environment:**
   ```
   conda activate your_env_name
   ```
   
   > To remove the environment in the future, use: `conda env remove --name your_env_name`

### Step 2: Install Project Dependencies

1. **Install required Python packages:**
   ```
   pip install -r requirements.txt
   ```

### Step 3: Create the .env File

1. **Generate a `.env` file in your project's root directory:**
   ```
   touch .env
   ```
   
2. **Add the necessary environment variables to the `.env` file:**
   ```
   echo "CHROMA_HOST=localhost" >> .env
   echo "CHROMA_PORT=1234" >> .env
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   ```

### Step 4: Run the Project

1. **Execute the application:**
   ```
   streamlit run chat_ui.py
   ```

### Data Preparation (Optional)

On the first run, the application will download data from Hugging Face (ARR CORP - SEC Return 2019 and 2020) and generate embeddings, which can take up to 15-20 minutes. To avoid this, you can manually download the embeddings from the provided Google Drive link:

1. **Download embeddings:**
   [Google Drive Folder](https://drive.google.com/drive/folders/1XF8NZEqZwBMXrM2X4_zx88B0_jyTMUbN?usp=share_link)

2. **Prepare the `chroma` folder:**
   - Unzip the downloaded file.
   - Create a folder named `chroma` at the root of your project.
   - Move the unzipped content into the `chroma` folder.

---

Ensure to replace placeholders like `your_env_name` and `your_openai_api_key_here` with your specific details. This guide is designed to help users through the setup process of the KnowledgeBridgeLLM project efficiently.

# Model Evaluation

To evaluate the model's performance, you need to execute the `evaluation.py` script. Below are the detailed instructions:

## Running the Evaluation Script

1. **Execute the Script**: Navigate to the directory containing `evaluation.py` and run the script in your command line interface:
    ```bash
    python evaluation.py
    ```

2. **Understanding the Process**: For a visual representation of the evaluation process, refer to the flowchart available in the `flow_diagram` folder.

3. **Test Outcome Data**: The repository includes a pre-compiled test outcome file named `testOutcome.csv`. This file contains the results of a previous evaluation run.

4. **Generating New Tests**: If you wish to generate a new test file, simply remove or rename the existing `testOutcome.csv` file in the repository. Note that generating new tests consumes a significant number of tokens since it utilizes large language models (LLMs) for backend processing.

## Note on Resource Consumption

Please be aware that running new tests is resource-intensive and may take some time to complete due to the usage of backend LLMs. Plan your resource allocation accordingly.

For any issues or further instructions, refer to the documentation or open an issue in the repository.

