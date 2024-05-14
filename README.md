# Context-Question-Answer Project

## Description
This project is designed to process a PDF file from a given URL, extract text from it, generate embeddings using BERT, search for sentences similar to a given question, and finally provide an answer to the question based on the content of the PDF.

The core components of this project are:
1. **PDF Processing**: Downloads and extracts text from PDF files.
2. **Text Embedding**: Uses BERT to create embeddings from the extracted text.
3. **Similarity Search**: Utilizes FAISS to find the most relevant sentences in the PDF text that match the user's question.
4. **Question Answering**: Employs LangChain and GPT4All to generate answers based on the context provided by the most relevant sentences.

The project leverages several state-of-the-art technologies:
- **BERT**: For generating embeddings from the PDF text.
- **FAISS**: For efficient similarity search.
- **LangChain**: For constructing the question-answering chain.
- **GPT4All**: For generating human-like answers based on the context and question.

## Model Used
The project uses the GPT4All model: `orca-2-7b.Q4_0.gguf`.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd context-question-answer
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the model file (`orca-2-7b.Q4_0.gguf`) from [GPT4All Model](https://gpt4all.io/models/gguf/orca-2-7b.Q4_0.gguf) and place it in the project root. If the model file is not present, it will be downloaded automatically.

## Usage

### Using the Python Script
To run the script and get an answer based on the content of a PDF file, use the following command:
```sh
python src/main.py --url "<PDF_URL>" --question "<your_question>" --device "<cpu_or_gpu>"

```
### Example
```
python src/main.py --url "https://media.pragprog.com/titles/ktuk/excerpts.pdf" --question "what is a widget?" --device cpu

```
## Using the Jupyter Notebook
You can also use the Jupyter Notebook provided in the `notebooks` directory. Open the notebook and fill in the necessary fields in the following function:
```
url = "your link here"
question = "your question here"
device_type = "cpu"

answer = main(url, question, device_type)
print("Answer:", answer)
