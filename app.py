import os
from flask import Flask, request, jsonify
from model import ModelLoader
from tokenizers import Tokenizer

app = Flask(__name__)

# 获取当前文件的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))

# 定义模型和配置文件的相对路径
model_path = os.path.join(base_path, 'model.pth')
config_path = os.path.join(base_path, 'config.json')
tokenizer_path = os.path.join(base_path, 'tokenizer.json')

# 加载模型和分词器
model_loader = ModelLoader(model_path=model_path, config_path=config_path)
tokenizer = Tokenizer.from_file(tokenizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query')
    history = data.get('history', [])
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 0.95)
    response = model_loader.predict(tokenizer, query, history, temperature, top_p)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)