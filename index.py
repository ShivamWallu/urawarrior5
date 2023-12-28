from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import os
from IPython.display import Markdown, display
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def get_response(query):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(query)
    return response.response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.get_json()
    query = data.get('query', '')
    response = get_response(query)
    return jsonify({"response": response})


if __name__ == '__main__':
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "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"

    # Construct the index
    construct_index("Context")

    # Run the Flask app
    app.run(host='0.0.0.0', port=80, debug=False)
