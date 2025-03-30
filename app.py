"""MegaLLM Flask 应用接口

提供基于HTTP的模型推理服务，支持:
- 文本生成
- 多模态输入
- 参数调节

API端点:
    POST /predict - 执行模型推理

运行方式:
    python app.py
"""

import os
from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
from model import ModelLoader
from tokenizers import Tokenizer
import torch
from flask import Response
import json

app = Flask(__name__)

# 获取当前文件的绝对路径
base_path = os.path.dirname(os.path.abspath(__file__))

# 定义模型和配置文件的相对路径
model_path = os.path.join(base_path, 'model.pth')
config_path = os.path.join(base_path, 'config.json')
tokenizer_path = os.path.join(base_path, 'tokenizer.json')

# 加载分词器并设置特殊token
tokenizer = Tokenizer.from_file(tokenizer_path)

# 定义特殊token
special_tokens = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "user_token": "<|user|>",
    "system_token": "<|system|>"
}

# 为tokenizer添加特殊token属性
for token_name, token in special_tokens.items():
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        print(f"警告: 特殊token {token} 不在词汇表中")
        token_id = len(tokenizer.get_vocab())
        tokenizer.add_tokens([token])
    setattr(tokenizer, token_name, token)
    setattr(tokenizer, f"{token_name}_id", token_id)

# 加载模型
model_loader = ModelLoader(
    # model_path=model_path, 
    config_path=config_path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    quantize=True
)

def process_base64_image(base64_string):
    """解码base64编码的图像
    
    Args:
        base64_string: base64编码的图像字符串
        
    Returns:
        Image: PIL图像对象
        
    Raises:
        ValueError: 如果解码失败
    """
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
    """处理预测请求
    
    Request JSON格式:
    {
        "query": "输入文本",
        "image": "base64图像(可选)",
        "history": ["历史对话"(可选)],
        "temperature": 温度值(可选),
        "top_p": top_p值(可选),
        "max_length": 最大长度(可选),
        "num_beams": beam数(可选)
    }
    
    Returns:
        JSON: 包含响应或错误信息
        
    HTTP状态码:
        200: 成功
        400: 请求参数错误
        500: 服务器内部错误
    """
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': '缺少查询文本'}), 400
            
        # 获取可选参数
        history = data.get('history', [])
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.95))
        max_length = int(data.get('max_length', 100)) if 'max_length' in data else None
        num_beams = int(data.get('num_beams', 1))
        
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
            top_p=top_p,
            max_length=max_length,
            num_beams=num_beams
        )
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream_predict', methods=['POST'])
def stream_predict():
    """流式预测接口
    
    Request JSON格式与/predict相同
    Response为SSE(Server-Sent Events)格式
    """
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': '缺少查询文本'}), 400
            
        # 获取其他参数
        history = data.get('history', [])
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.95))
        max_length = int(data.get('max_length', 100)) if 'max_length' in data else None
        num_beams = int(data.get('num_beams', 1))
        
        # 处理图像
        image = None
        if 'image' in data:
            image = process_base64_image(data['image'])
            
        def generate():
            # 流式生成响应
            for chunk in model_loader.stream_predict(
                tokenizer=tokenizer,
                query=query,
                image=image,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length,
                num_beams=num_beams
            ):
                yield f"data: {json.dumps({'response': chunk})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    """启动Flask应用"""
    app.run(host='0.0.0.0', port=5000, threaded=True)