
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import socket
import time
import json
import LayoutLLM

current_ip = socket.gethostbyname(socket.gethostname())
app = Flask(__name__)

layout_gpt = LayoutLLM.FewShotLearningGPT()
layout_claude3 = LayoutLLM.FewShotLearningClaude3()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gen_layout', methods=['POST'])
def gen_layout():
    # define output
    output_json = {'status':'fail',
                   'error_reason':''}
    # acquire input question from frontend
    json_str = request.data.decode('utf-8')
    json_data = json.loads(json_str)
    query = json_data['query']
    print(f"User Query: {query}")
    t_start = time.time()
    try:
        ####### generate layout by chatgpt
        # prompt, generation_text = layout_gpt.generate_layout_json(query,model='gpt4')
        # print(generation_text)
        # gen_str_data, gen_data = layout_gpt.catch_json_from_generation(generation_text)
        # print(gen_data)
        # img_base64 = layout_gpt.plot_data(data = gen_data, 
        #                                     plot = False)
        
        ###### generate layout by claude3
        prompt, generation_text = layout_claude3.generate_layout_json(query)
        print(generation_text)
        gen_str_data, gen_data = layout_claude3.catch_json_from_generation(generation_text)
        print(gen_data)  
        img_base64 = layout_claude3.plot_data(data = gen_data, 
                                            plot = False)
    except Exception as e:
        output_json['error_reason'] = f'[Error] {str(e)}'
        print(output_json['error_reason'])
    else:
        output_json['status'] = 'success'
        output_json['img_base64'] = img_base64
        output_json['gen_str'] = gen_str_data
        output_json['prompt_str'] = prompt
    t_end = time.time()
    print(f'Time cost: {t_end - t_start} s')
    
    return jsonify(output_json)



@app.route('/fix_layout', methods=['POST'])
def fix_layout():
    # define output
    output_json = {'status':'fail',
                   'error_reason':''}
    # acquire input question from frontend
    json_str = request.data.decode('utf-8')
    json_data = json.loads(json_str)
    query = json_data['query']
    history_messages = json_data['history_messages']
    print(f"User Query: {query}")
    t_start = time.time()
    try:        
        ###### generate layout by claude3
        query, generation_text = layout_claude3.optimize_layout(history_messages , query)
        print(generation_text)
        gen_str_data, gen_data = layout_claude3.catch_json_from_generation(generation_text)
        print(gen_data)  
        img_base64 = layout_claude3.plot_data(data = gen_data, 
                                            plot = False)
    except Exception as e:
        output_json['error_reason'] = f'[Error] {str(e)}'
        print(output_json['error_reason'])
    else:
        output_json['status'] = 'success'
        output_json['img_base64'] = img_base64
        output_json['gen_str'] = gen_str_data
        output_json['prompt_str'] = query
    t_end = time.time()
    print(f'Time cost: {t_end - t_start} s')
    
    return jsonify(output_json)



if __name__ == '__main__':
    app.run(host=current_ip,port=4949,debug=True)