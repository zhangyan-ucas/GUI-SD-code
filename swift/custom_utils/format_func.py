import json 
import re

def extract_action(content):
    try:
        output_text = json.loads(content.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

        if 'arguments' not in output_text or 'action' not in output_text['arguments']:
            return "no action"
        return output_text
    except:
        return "no action"


def cold_extract_action(content):
    try:
        output_text = eval(content.split('<tool_call>\n')[-1].split('\n</tool_call>')[0])

        if 'arguments' not in output_text or 'action' not in output_text['arguments']:
            return "no action"
        return output_text
    except:
        return "no action"

def extract_ground(text):
    json_match = re.search(r"```json(.*?)```", text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            data = json.loads(json_str)
            point = data[0]['point_2d']
            return point
            
        except:
            return "no action"
    else:
        return "no action"