#-*-coding:utf8-*-
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from model import Model
# from gevent.wsgi import WSGIServer
from gevent import monkey
monkey.patch_all()
from gevent.pywsgi import WSGIServer
import logging

app = Flask(__name__)
CORS(app)

vocab_file = './data/vocabs'
model_dir = './models/output_couplet'

m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)

@app.route('/connect')
def connect():
    return "conncect success!"


# @app.route('/chat/couplet/<in_str>', methods=['GET', 'POST'])
@app.route('/chat/couplet', methods=['POST'])
def chat_couplet():
    input = request.form["input"]
    if len(input) == 0 or len(input) > 50:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(input))
        output = ''.join(output.split(' '))
    # print('上联：%s；下联：%s' % (in_str, output))
    print("上联:{},下联:{}".format(input, output))
    return jsonify({'input':input,'output': output})


http_server = WSGIServer(('0.0.0.0', 8096), app)
http_server.serve_forever()
