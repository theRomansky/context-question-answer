import os
import re
import requests
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import faiss
import torch
from transformers import BertTokenizer, BertModel
from gpt4all import GPT4All
from langchain_community.llms import GPT4All as LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Get the absolute path to the project's root directory
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
model_directory = os.path.join(project_root, 'model')


def download_pdf_from_url(url, save_path):
    """
    Download a PDF file from the given URL and save it to the specified path.

    :param url: The URL of the PDF file.
    :param save_path: The path where the PDF file should be saved.
    """
    response = requests.get(url)
    with open(save_path, 'wb') as out_file:
        out_file.write(response.content)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    :param pdf_path: The path to the PDF file.
    :return: The extracted text from the PDF file.
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()
    os.remove(pdf_path)

    if text:
        return text


def preprocess_text(text):
    """
    Preprocess the text by removing extra whitespaces and splitting it into sentences.

    :param text: The text to be processed.
    :return: A list of sentences derived from the text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return sentences


def embed_sentences(sentences, use_gpu=False):
    """
    Embed sentences using the BERT model.

    :param sentences: A list of sentences to be embedded.
    :param use_gpu: A boolean indicating whether to use GPU for embedding or not. Default is False.
    :return: A tensor containing the embedded representations of the sentences.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', resume_download=False)
    model.to(device)

    tokenized_texts = [tokenizer.encode(sentence, add_special_tokens=True, max_length=512, truncation=True) for sentence in sentences]
    max_len = max(len(sent) for sent in tokenized_texts)
    padded_tokenized_texts = [sent + [tokenizer.pad_token_id] * (max_len - len(sent)) for sent in tokenized_texts]

    indexed_tokens = torch.tensor(padded_tokenized_texts).to(device)
    with torch.no_grad():
        outputs = model(input_ids=indexed_tokens)
        encoded_layers = outputs.last_hidden_state

    embeddings = encoded_layers[:, 0, :]
    return embeddings


def create_faiss_index(embeddings):
    """
    Create a FAISS index with the given embeddings.

    :param embeddings: The embeddings to be indexed.
    :return: The created FAISS index.
    """
    index = faiss.IndexFlatL2(embeddings.size(-1))
    index.add(embeddings.cpu().numpy())
    return index


def download_model(model_url, model_directory, model_filename):
    """
    Download a model from the given URL and save it to the specified directory.

    :param model_url: The URL of the model to download.
    :param model_directory: The directory to save the downloaded model.
    :param model_filename: The filename to save the downloaded model as.
    """
    model_path = os.path.join(model_directory, model_filename)
    if os.path.exists(model_path):
        print("Model found")
    else:
        print("Model not found. Starting download...")
        os.makedirs(model_directory, exist_ok=True)
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("Model successfully downloaded and saved.")
        else:
            print("Failed to download model.")


def create_langchain(question, context, model_path):
    """
    Create a LangChain LLMChain to generate an answer based on the question and context.

    :param question: The question to be answered.
    :param context: The context related to the question.
    :param model_path: The path to the model file.
    :return: The generated answer.
    """
    prompt_template = """Context: {context}\n\nQuestion: {question}\n\nAnswer:"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = LLM(model=model_path)
    chain = LLMChain(prompt=prompt, llm=llm)
    answer = chain.run(context=context, question=question)
    return answer


def find_similar_sentences_in_pdf(question, pdf_text, index, use_gpu=False):
    """
    Find the most similar sentence in a PDF based on a given question.

    :param question: The question to compare sentences against.
    :param pdf_text: The text content of the PDF.
    :param index: The FAISS index used for similarity search.
    :param use_gpu: A boolean indicating whether to use GPU for embeddings. Default is False.
    :return: The most similar sentence to the question in the PDF.
    """
    question_embedding = embed_sentences([question], use_gpu)
    D, I = index.search(question_embedding.cpu().numpy(), k=1)
    most_similar_index = I[0][0]
    most_similar_sentence = pdf_text[most_similar_index]
    return most_similar_sentence


def main(url, question, device_type):
    """
    Main function to download a PDF, extract text, create embeddings, search for similar sentences,
    and generate an answer using LangChain.

    :param url: The URL of the PDF file.
    :param question: The question to be answered.
    :param device_type: The type of device to use for embeddings (either 'cpu' or 'gpu').
    :return: The generated answer.
    """
    if device_type == "cpu":
        use_gpu = False
    elif device_type == "gpu":
        use_gpu = True
    else:
        raise ValueError("Invalid value for device_type flag. Please provide either 'cpu' or 'gpu'.")

    model_url = "https://gpt4all.io/models/gguf/orca-2-7b.Q4_0.gguf"
    model_directory = "../"
    model_filename = "orca-2-7b.Q4_0.gguf"
    model_path = os.path.join(model_directory, model_filename)
    pdf_save_path = "temp.pdf"

    download_pdf_from_url(url, pdf_save_path)
    download_model(model_url, model_directory, model_filename)
    pdf_text = extract_text_from_pdf(pdf_save_path)
    pdf_preprocessed_text = preprocess_text(pdf_text)
    embeddings = embed_sentences(pdf_preprocessed_text, use_gpu)
    index = create_faiss_index(embeddings)
    context = find_similar_sentences_in_pdf(question, pdf_preprocessed_text, index, use_gpu)
    answer = create_langchain(question, context, model_path)

    return answer


if __name__ == "__main__":
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process a PDF file and answer a question based on its content.")

    # Add the URL argument for the PDF file
    parser.add_argument("--url", type=str, required=True, help="The URL of the PDF file.")

    # Add the question argument that needs to be answered
    parser.add_argument("--question", type=str, required=True, help="The question to be answered.")

    # Add the device type argument to specify whether to use CPU or GPU
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu", help="The device type to use (cpu or gpu).")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main function with the provided arguments and print the answer
    answer = main(args.url, args.question, args.device)
    print("Answer:", answer)