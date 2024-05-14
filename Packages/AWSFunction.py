# -*- coding: utf-8 -*-
import json
import time
import boto3



def chat_completion_anthropic(prompt, 
                              model_id="anthropic.claude-3-opus-20240229-v1:0"):
    # re-define output
    return_json={
                'prompt_tokens': 0,
                'completion_tokens':0,
                'total_tokens': 0,
                'status':"fail",
                'selected_model':model_id,
                'replied_message':"",
                'error_reason':""
            }
    
    # init the Amazon Bedrock runtime client
    client = boto3.client(service_name="bedrock-runtime", 
                        region_name="us-west-2")
    try:
        # invoke claude3 with the text prompt
        response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4096,
                        "temperature":1.0,
                        "messages":[{
                                    "role":"user",
                                    "content":[{"type":"text","text":prompt}]
                                }
                            ]
                        }
                    )
            )
        # Process the response
        result = json.loads(response["body"].read())
        
        return_json['prompt_tokens'] = result['usage']['input_tokens']
        return_json['completion_tokens'] = result['usage']['output_tokens']
        return_json['total_tokens'] = return_json['prompt_tokens'] + return_json['completion_tokens']
        return_json['replied_message'] = result['content'][0]['text']
        return_json['status'] = "success"
    except Exception as e:
        error_reason = "[BEDROCK ERROR]"+str(e)
        return_json["error_reason"] = error_reason
        print(error_reason)
        
    return return_json



# history_messages for claude3
def convert2HistoryMessages(history_messages, query):
    llm_history_messages = []
    for i, pair in enumerate(history_messages,1):
        Qi, Ai = pair
        llm_history_messages.append({"role":"user","content":[{"type":"text","text":Qi}]})
        llm_history_messages.append({"role": "assistant", "content": [{"type":"text","text":Ai}]})
    llm_history_messages.append({"role":"user","content":[{"type":"text","text":query}]})
    return llm_history_messages


def chat_completion_anthropic_history(llm_history_messages:list=[], 
                                      model_id="anthropic.claude-3-opus-20240229-v1:0"):
    # re-define output
    return_json={
                'prompt_tokens': 0,
                'completion_tokens':0,
                'total_tokens': 0,
                'status':"fail",
                'selected_model':model_id,
                'replied_message':"",
                'error_reason':""
            }
    
    # init the Amazon Bedrock runtime client
    client = boto3.client(service_name="bedrock-runtime", 
                        region_name="us-west-2")
    try:
        # invoke claude3 with the text prompt
        response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4096,
                        "temperature":1.0,
                        "messages":llm_history_messages
                        }
                    )
            )
        # Process the response
        result = json.loads(response["body"].read())
        
        return_json['prompt_tokens'] = result['usage']['input_tokens']
        return_json['completion_tokens'] = result['usage']['output_tokens']
        return_json['total_tokens'] = return_json['prompt_tokens'] + return_json['completion_tokens']
        return_json['replied_message'] = result['content'][0]['text']
        return_json['status'] = "success"
    except Exception as e:
        error_reason = "[BEDROCK ERROR]"+str(e)
        return_json["error_reason"] = error_reason
        print(error_reason)
        
    return return_json




if __name__ == '__main__':
    print(chat_completion_anthropic(prompt="Hi"))

