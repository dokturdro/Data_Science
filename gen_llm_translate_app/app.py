import os
from argparse import ArgumentParser
from functools import lru_cache
import time

import flask
from flask import Flask, request, send_from_directory, Blueprint
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from globalvars import MODEL, LANGS

device = torch.device(torch.cuda.is_available() and 'cuda' or 'cpu')

DEF_SRC_LNG = 'eng_Latn'
DEF_TGT_LNG = 'pol_Latn'

app = Flask(__name__)
bp = Blueprint('nmt', __name__, template_folder='templates', static_folder='static')

def render_template(*args, **kwargs):
    return flask.render_template(*args, environ=os.environ, **kwargs)

def jsonify(obj):

    if obj is None or isinstance(obj, (int, bool, str)):
        return obj
    elif isinstance(obj, dict):
        return {key: jsonify(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [jsonify(it) for it in obj]

def attach_translate_route(
    model_id=MODEL, def_src_lang=DEF_SRC_LNG,
    def_tgt_lang=DEF_TGT_LNG, **kwargs):
    torch.set_grad_enabled(False)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # src_langs = tokenizer.additional_special_tokens
    src_langs = [k for k, v in LANGS.items()]
    tgt_langs = src_langs

    @lru_cache(maxsize=256)
    def get_tokenizer(src_lang=def_src_lang):
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        return AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)

    @bp.route('/')
    def index():
        args = dict(src_langs=src_langs, tgt_langs=tgt_langs, model_id=model_id,
                    def_src_lang=def_src_lang, def_tgt_lang=def_tgt_lang)
        return render_template('index.html', **args)


    @bp.route("/translate", methods=["POST", "GET"])
    def translate():
        st = time.time()
        if request.method not in ("POST", "GET"):
            return "GET and POST are supported", 400
        if request.method == 'GET':
            args = request.args
        if request.method == 'POST':
            if request.headers.get('Content-Type') == 'application/json':
                args = request.json
            else:
                args = request.form

        sources = args.get("source")
        src_lang = args.get('src_lang') or def_src_lang
        tgt_lang = args.get('tgt_lang') or def_tgt_lang
                        
        tokenizer = get_tokenizer(src_lang=src_lang)

        max_length = 80
        inputs = tokenizer(sources, return_tensors="pt", padding=True)
        inputs = {k:v.to(device) for k, v in inputs.items()}

        translated_tokens = model.generate(
            **inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length = max_length)
        results = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        res = dict(source=sources, translation=results,
                   src_lang = src_lang, tgt_lang=tgt_lang)

        return flask.jsonify(jsonify(res))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run Flask server in debug mode")
    parser.add_argument("-p", "--port", type=int, help="port to run server on", default=8000)
    parser.add_argument("-ho", "--host", help="Host address to bind.", default='0.0.0.0')
    args = vars(parser.parse_args())
    return args


cli_args = parse_args()
attach_translate_route(**cli_args)
app.register_blueprint(bp, url_prefix=cli_args.get('base'))

def main():
    app.run(debug=cli_args["debug"], port=cli_args["port"], host=cli_args["host"])

if __name__ == "__main__":
    main()