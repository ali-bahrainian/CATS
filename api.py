from flask import Flask, request
from encode import write_to_bin
import uuid
import os
import shutil

app = Flask(__name__)
app.config["DEBUG"] = False

@app.route('/', methods=['GET'])
def home():
    return "<h1>CATS: Customizable Abstractive Topic-based Summarization</h1><p>This site is the interface of an API to interact with an advanced topic-aware summarization model.</p>"

@app.route('/summarize', methods=['POST'])
def get_summary():
    # Define settings
    TARGET_INPUT_FOLDER = 'requests'
    TRAINED_MODEL_FOLDER = 'logs'
    TRAINED_MODEL_EXPERIMENT = 'cats-train-full'

    # Generate other variables
    request_uuid = request_uuid = str(uuid.uuid4())
    input_file_location = os.path.join(TARGET_INPUT_FOLDER, request_uuid, 'test_000.bin')

    # Get target text
    input_text = request.form.get('target')

    # Write target to binary for decoding
    write_to_bin(input_text, input_file_location)

    # Create summary
    os.system(f"python3 run_summarization.py --mode=decode --data_path='{os.path.join(TARGET_INPUT_FOLDER, request_uuid)}/test_*' --vocab_path='data/vocab' --log_root='{TRAINED_MODEL_FOLDER}' --exp_name={TRAINED_MODEL_EXPERIMENT} --single_pass=True --single_input=True --decode_dir='{os.path.join('requests', request_uuid)}'")

    # Read the summary file
    with open(os.path.join(TARGET_INPUT_FOLDER, request_uuid, 'decoded', '000000_decoded.txt')) as f:
        summary = f.read().splitlines() 

    # Delete the request directory
    shutil.rmtree(os.path.join(TARGET_INPUT_FOLDER, request_uuid))

    # Return the summary
    return {
        'summary': summary
    }


app.run()