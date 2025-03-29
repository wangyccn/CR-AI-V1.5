import os
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
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

def process_base64_image(base64_string):
    try:
        # 移除base64头信息
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码base64字符串
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"图像处理失败: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': '缺少查询文本'}), 400
            
        # 获取可选参数
        history = data.get('history', [])
        temperature = data.get('temperature', 1.0)
        top_p = data.get('top_p', 0.95)
        
        # 处理图像（如果有）
        image = None
        if 'image' in data:
            image = process_base64_image(data['image'])
            
        response = model_loader.predict(
            tokenizer=tokenizer,
            query=query,
            image=image,
            history=history,
            temperature=temperature,
            top_p=top_p
        )
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)