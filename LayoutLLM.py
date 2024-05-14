# -*- coding: utf-8 -*-
import sys
sys.path.append("Packages")
import json
import re
import base64
import time
from io import BytesIO
import OpenAIFunction
import AWSFunction
import matplotlib.pyplot as plt


class FewShotLearningGPT:
    def __init__(self) -> None:
        # regex
        self.pattern = r'```(.*?)```'
        # this prompt is about 4500 tokens
        self.fewShotPrompt = """Here are example json files that represents a EBPro user interface and can be described in the following ways:  
                                1."The group consists of two TEXT_OBJECT elements and one NUMERICDATA_OBJECT. To show a measure on screen, usually three elements are shown together: label, unit and value"  
                                {  
                                    "resolution_x": 1024,  
                                    "resolution_y": 768,  
                                    "window_info": {  
                                        "height": 600,  
                                        "width": 1024,  
                                        "object_infos": [  
                                            {  
                                                "height": 17,  
                                                "pos_x": 371,  
                                                "pos_y": 263,  
                                                "type": 21,  
                                                "type_name": "TEXT_OBJECT",  
                                                "width": 113,  
                                                "description":"label of a measure like temperature"  
                                            },  
                                            {  
                                                "height": 19,  
                                                "pos_x": 593,  
                                                "pos_y": 264,  
                                                "type": 21,  
                                                "type_name": "TEXT_OBJECT",  
                                                "width": 19,  
                                                "description":"unit label of a measure like °C"  
                                            },  
                                            {  
                                                "height": 18,  
                                                "pos_x": 527,  
                                                "pos_y": 264,  
                                                "type": 30,  
                                                "type_name": "NUMERICDATA_OBJECT",  
                                                "width": 64,  
                                                "description":"numeric input or display for a measure"  
                                            }  
                                        ]  
                                    },  
                                    "group_info": {  
                                        "group_width": 241,  
                                        "group_height": 20,  
                                        "group_pos_x": 371,  
                                        "group_pos_y": 263,  
                                        "description":"The group consists of two TEXT_OBJECT elements and one NUMERICDATA_OBJECT. To show a measure on screen, usually three elements are shown together: label, unit and value"  
                                    }  
                                }  
                                2. "The group is used to display a tank with scale, liquid level in 2D"
                                {
                                    "resolution_x": 0,
                                    "resolution_y": 0,
                                    "window_info": {
                                        "height": 768,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 310,
                                                "pos_x": 675,
                                                "pos_y": 279,
                                                "type": 22,
                                                "type_name": "GRAPH_OBJECT",
                                                "width": 196,
                                                "description":"A tank image"
                                            },
                                            {
                                                "height": 135,
                                                "pos_x": 793,
                                                "pos_y": 356,
                                                "type": 42,
                                                "type_name": "SHAPE_OBJECT",
                                                "width": 29,
                                                "description":"background of liquid level which indicates the inner-tank color, usually gray"
                                            },
                                            {
                                                "height": 133,
                                                "pos_x": 794,
                                                "pos_y": 357,
                                                "type": 37,
                                                "type_name": "BARGRAPH_OBJECT",
                                                "width": 27,
                                                "description":"a liquid level in the tank. A single bar to represents increasing or decreasing liquid level"
                                            },
                                            {
                                                "height": 134,
                                                "pos_x": 825,
                                                "pos_y": 356,
                                                "type": 83,
                                                "type_name": "DYNAMICSCALE_OBJECT",
                                                "width": 30,
                                                "description":"scale of liquid level (e.g. 0%~100%)"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 254,
                                        "group_height": 286,
                                        "group_pos_x": 690,
                                        "group_pos_y": 323,
                                        "description":"The group is used to display a tank with scale, liquid level in 2D"
                                    }
                                }
                                3. "The group consists of six numerics showing YEAR, MONTH, DAY, HOUR, MINUTE and SECOND. To make display more clear, using '/' separator to distinguish YEAR/MONTH/DAY. Units of HOUR, MINUTE and SECOND are also considered"
                                {
                                    "resolution_x": 0,
                                    "resolution_y": 0,
                                    "window_info": {
                                        "height": 768,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 39,
                                                "pos_x": 235,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for YEAR"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 140,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for MONTH"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 45,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for DAY"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 45,
                                                "pos_y": 345,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 46,
                                                "description":"A text to describe DAY/MONTH/YEAR, usually called Date"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 130,
                                                "pos_y": 380,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 7,
                                                "description":"a text like '/' to separate DAY and MONTH"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 225,
                                                "pos_y": 380,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 7,
                                                "description":"a text like '/' to separate MONTH and YEAR"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 255,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for SECOND"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 139,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for MINUTE"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 45,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for HOUR"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 45,
                                                "pos_y": 425,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 57,
                                                "description":"a text to describe HH:MM:SS, usually called Time"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 126,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 12,
                                                "description":"a text unit for HOUR, use 'h' to represent the unit"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 220,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 33,
                                                "description":"a text unit for MINUTE, use 'min' to represent the unit"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 335,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 12,
                                                "description":"a text unit for SECOND, use 's' to represent the unit"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 302,
                                        "group_height": 147,
                                        "group_pos_x": 45,
                                        "group_pos_y": 345,
                                        "description":"The group consists of six numerics showing YEAR, MONTH, DAY, HOUR, MINUTE and SECOND. To make display more clear, using '/' separator to distinguish YEAR/MONTH/DAY. Units of HOUR, MINUTE and SECOND are also considered"
                                    }
                                }
                                4. "A guage or meter display to show a value from range 0-21000 kicks/hr. The group contains not only a meter, but also its numeric value and unit text"
                                {
                                    "resolution_x": 1920,
                                    "resolution_y": 1080,
                                    "window_info": {
                                        "height": 1080,
                                        "width": 1920,
                                        "object_infos": [
                                            {
                                                "height": 719,
                                                "pos_x": 627,
                                                "pos_y": 201,
                                                "type": 70,
                                                "type_name": "METERDISPLAY_CMT_OBJECT",
                                                "width": 719,
                                                "description":"A meter display with min=0 and max=21000"
                                            },
                                            {
                                                "height": 782,
                                                "pos_x": 597,
                                                "pos_y": 165,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 782,
                                                "description":"The bitmap acts as the background of meter. It can be also a monitor on a plc/hmi address"
                                            },
                                            {
                                                "height": 238,
                                                "pos_x": 696,
                                                "pos_y": 807,
                                                "type": 4,
                                                "type_name": "RECTANGLE_OBJECT",
                                                "width": 614,
                                                "description":"For purpose of beautification , a rectangle that wraps a numeric and a text to get user good understanding of meter value and its unit"
                                            },
                                            {
                                                "height": 140,
                                                "pos_x": 747,
                                                "pos_y": 821,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 514,
                                                "description":"numeric display wrapped in a rectangle for meter value"
                                            },
                                            {
                                                "height": 80,
                                                "pos_x": 879,
                                                "pos_y": 954,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 263,
                                                "description":"A text showing the unit of numeric display and wrapped in a rectangle"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 719,
                                        "group_height": 844,
                                        "group_pos_x": 627,
                                        "group_pos_y": 201,
                                        "group_name": "object_detail_MainScreen_group_5.png",
                                        "description":"A guage or meter display to show a value from range 0-21000 kicks/hr. The group contains not noly a meter, but also its numeric value and unit text"
                                    }
                                }
                                5. "The group represents a manual which contains two buttons (actually function keys) to achieve page switch."
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 69,
                                                "pos_x": 62,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the leftmost separator for function key"
                                            },
                                            {
                                                "height": 69,
                                                "pos_x": 162,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the middle separator to separate two function keys"
                                            },
                                            {
                                                "height": 69,
                                                "pos_x": 262,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the rightmost separator for function key"
                                            },
                                            {
                                                "height": 71,
                                                "pos_x": 80,
                                                "pos_y": 512,
                                                "type": 33,
                                                "type_name": "FUNKEY_OBJECT",
                                                "width": 71,
                                                "description":"The function key is used to page switch. In this case, page will be switched to 'home' as it is pressed"
                                            },
                                            {
                                                "height": 68,
                                                "pos_x": 176,
                                                "pos_y": 518,
                                                "type": 33,
                                                "type_name": "FUNKEY_OBJECT",
                                                "width": 74,
                                                "description":"The function key is used to page switch. In this case, page will be switched to 'Window 17' as it is pressed"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 201,
                                        "group_height": 74,
                                        "group_pos_x": 62,
                                        "group_pos_y": 512,
                                        "description":"The group represents a manual which contains two buttons (actually function keys) to achieve page switch."
                                    }
                                }
                                6. "The group consists of a bit lamp and a toggle switch. Here the toggle changes value of an address, and the bit lamp acts as a text object to describe its toggle"
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 69,
                                                "pos_x": 203,
                                                "pos_y": 249,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 195,
                                                "description":"This bit lamp is monitoring on an PLC address, and value of the address will affects state of the bimap. This bit lamp here acts as a text object to describe a toggle button as well"
                                            },
                                            {
                                                "height": 99,
                                                "pos_x": 84,
                                                "pos_y": 236,
                                                "type": 27,
                                                "type_name": "TOGGLESWITCH_OBJECT",
                                                "width": 99,
                                                "description":"The toggle switch (or called toggle button) can change value of an address, leading to the state change of a bit lamp"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 314,
                                        "group_height": 99,
                                        "group_pos_x": 84,
                                        "group_pos_y": 236,
                                        "description":"The group consists of a bit lamp and a toggle switch. Here the toggle changes value of an address, and the bit lamp acts as a text object to describe its toggle."
                                    }
                                }
                                7. "This group use a grid object as a table which size is 5 X 2. It completely contains current info of a running machine"
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 211,
                                                "pos_x": 275,
                                                "pos_y": 380,
                                                "type": 85,
                                                "type_name": "GRID_OBJECT",
                                                "width": 570,
                                                "description":"The grid object acts as a table with 5 X 2 cells in this case. Then, user can drag any object into the cells where these objects are automatically arranged like an array"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 662,
                                                "pos_y": 512,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 80,
                                                "description":"The numeric object shows total batch weight of a machine. It's located at (4,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 328,
                                                "pos_y": 515,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 179,
                                                "description":"The text object describes batch weight of a machine. its content is 'Total Batch weight'. It's located at (4,1) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 348,
                                                "pos_y": 389,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 138,
                                                "description":"The text object describes state of a machine. its content is 'Batching State'. It's located at (1,1) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 582,
                                                "pos_y": 389,
                                                "type": 24,
                                                "type_name": "WORDLAMP_OBJECT",
                                                "width": 240,
                                                "description":"Unlike bit lamp (only ON/OFF), The word lamp object can show multi-state of a machine. It's located at (1,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 361,
                                                "pos_y": 557,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 113,
                                                "description":"The text object describes Fault Status of a machine. its content is 'Fault Status'. It's located at (5,1) of the grid object table"
                                            },
                                            {
                                                "height": 37,
                                                "pos_x": 665,
                                                "pos_y": 551,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 74,
                                                "description":"The bit lamp shows Fault Status of a machine. It's located at (5,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 346,
                                                "pos_y": 431,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 142,
                                                "description":"The text object describes recipe of a machine. its content is 'Current Recipe'. It's located at (2,1) of the grid object table"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 573,
                                                "pos_y": 428,
                                                "type": 32,
                                                "type_name": "ASCIIDATA_OBJECT",
                                                "width": 259,
                                                "description":"The ascii object is aimed to show the current recipe value. It's located at (2,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 346,
                                                "pos_y": 473,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 142,
                                                "description":"The text object describes weight of a machine. its content is 'Current Weight'. It's located at (3,1) of the grid object table"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 662,
                                                "pos_y": 470,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 80,
                                                "description":"The numeric object shows current weight of a machine. It's located at (3,2) of the grid object table"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 570,
                                        "group_height": 211,
                                        "group_pos_x": 275,
                                        "group_pos_y": 380,
                                        "description":"This group use a grid object as a table which size is 5 X 2. It completely contains current info of a running machine"
                                    }
                                }

                                In addition to the elements in the above, here is a list of permitted objects in EBPro for UI elements:
                                PIE_OBJECT, ELLIPSE_OBJECT, ARC_OBJECT, POLYGON_OBJECT, ANIMATION_OBJECT, VIDEO_OBJECT, ASCIIINPUT_OBJECT, QR_CODE_OBJECT, FILE_BROWSER_OBJECT,
                                PICTURE_VIEWER_OBJECT, ALARMDISPLAY_OBJECT, XYPLOT_OBJECT, GRID_OBJECT, SETWORD_OBJECT, COMBO_BUTTON_OBJECT, OPTIONLIST_OBJECT, WORDLAMP_OBJECT,
                                MULTISTATESWITCH_OBJECT, INDIRECTWINDOW_OBJECT, DIRECTWINDOW_OBJECT, TIMER_OBJECT, TRENDDISPLAY_OBJECT, RECIPEVIEW_OBJECT, PDF_READER_OBJECT,
                                LINE_OBJECT, SLIDER_OBJECT


                                please generate another json file in the above format, that represents a UI with the following rules and description:
                                rule1: You should fully define the layout pattern and cannot simplify your result.
                                rule2: Do not explain any words and add any annotation in your json file.
                                rule3: The type of object you generate must not exceed the permitted objects and examples
                                rule4: Try to generate at least 3-word description of each element in English

                                The description:  
                                """
        # define color and type
        self.type_color = {'TEXT_OBJECT': 'green', 'NUMERICDATA_OBJECT': 'blue', 'GRAPH_OBJECT': 'red',
                            'NUMERICINPUT_OBJECT':'cyan','METERDISPLAY_CMT_OBJECT':'pink','BITLAMP_OBJECT':'yellow',
                            'RECTANGLE_OBJECT':'gray','FUNKEY_OBJECT':'gold','TOGGLESWITCH_OBJECT':'purple',
                            'PIE_OBJECT':'olive','XYPLOT_OBJECT':'teal','QR_CODE_OBJECT':'brown',
                            'ELLIPSE_OBJECT':'lime','ARC_OBJECT':'turquoise','POLYGON_OBJECT':'deepskyblue',
                            'ANIMATION_OBJECT':'silver','VIDEO_OBJECT':'peru','ASCIIINPUT_OBJECT':'m',
                            'FILE_BROWSER_OBJECT':'orange','PICTURE_VIEWER_OBJECT':'deeppink','ALARMDISPLAY_OBJECT':'magenta',
                            'GRID_OBJECT':'greenyellow','SETWORD_OBJECT':'slategray','COMBO_BUTTON_OBJECT':'darkslategray',
                            'OPTIONLIST_OBJECT':'wheat','WORDLAMP_OBJECT':'dodgerblue','ASCIIDATA_OBJECT':'darkred',
                            'OPTIONLIST_OBJECT':'tomato','PDF_READER_OBJECT':'cornflowerblue','LINE_OBJECT':'magenta',
                            'BARGRAPH_OBJECT':'steelblue','SHAPE_OBJECT':'y','OTHER_OBJECT':'black'}
    def generate_layout_json(self, query , model = "gpt4"):
        prompt = self.fewShotPrompt + query
        replied_message = ''
        max_try = 3
        try_cnt = 0
        while try_cnt < max_try:
            try:
                resp = OpenAIFunction.chat_completion_openai(message=prompt, 
                                                             model=model,
                                                             max_token = 4096)
                status = resp["status"]
                if status == 'success':
                    replied_message = resp['replied_message']
                    # print(replied_message)
                    break
                else:
                    print(resp["error_reason"])
                    try_cnt += 1
            except Exception as e:
                print(str(e))
                try_cnt += 1
        return prompt, replied_message
    
    def catch_json_from_generation(self, result):
        matches = re.findall(self.pattern, result, re.DOTALL)
        #有捕捉到
        if len(matches) > 0:
            json_str_data = matches[0].replace("```", "").replace("json", "").replace("\n", "").replace("  ","")
        # 沒捕捉到
        else:
            json_str_data = result.replace("\n", "").replace("  ","")
        try:
            json_data = json.loads(json_str_data)
            # print(json_data)
            return json_str_data, json_data
        except Exception as e:
            print(str(e))
            return
        
    def plot_data(self,data,plot=True):
        plt.figure(figsize=(10,6))
        # load data
        window_width = data["window_info"]["width"]
        window_height = data["window_info"]["height"]
        object_infos = data["window_info"]["object_infos"]
        # plot template box
        plt.gca().add_patch(plt.Rectangle((0, 0), window_width, window_height,
                                        fill=False, edgecolor='black'))
        
        type_set = []
        for obj in object_infos:
            posX = obj["pos_x"]
            posY = obj["pos_y"]
            width = obj["width"]
            height = obj["height"]
            type_name = obj["type_name"]
            type_set.append(type_name)
            fill = True
            alpha = 0.3 if type_name in ['RECTANGLE_OBJECT','GRID_OBJECT','PICTURE_VIEWER_OBJECT'] else 0.3
            try:
                color = self.type_color[type_name]
            except Exception as e:
                color = self.type_color['OTHER_OBJECT']
            plt.gca().add_patch(plt.Rectangle((posX, posY), width, height,
                                            fill=fill, edgecolor='white', facecolor=color, alpha=alpha))
            
        
        spaceby = 10
        for type_name in set(type_set):
            try:
                color = self.type_color[type_name]
            except Exception as e:
                color = self.type_color['OTHER_OBJECT']
            # plot object box
            plt.gca().add_patch(plt.Rectangle((window_width+10, spaceby), 30, 20,
                                                fill=True, edgecolor=color, facecolor=color, alpha = 0.3))
            # plot text
            plt.text(window_width+50, spaceby+20, type_name, fontsize=6, color='black')
            spaceby += 50
        
        # set axis range
        plt.xlim(0, window_width+100)
        plt.ylim(0, window_height+100)
        # 設定座標軸位置為左上方
        plt.gca().set_ylim(plt.ylim()[::-1])
        # disable axis ticks
        plt.axis('off')
        if plot:
            plt.show()
            return
        else:
            # convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_base64  = base64.b64encode(buffer.read()).decode()
            return img_base64 
    

class FewShotLearningClaude3(FewShotLearningGPT):
    def __init__(self):
        # call parent and do init
        super().__init__()
        # re-define step prompt
        self.fewShotPrompt="""Here are example json files that represents a EBPro user interface and can be described in the following ways:  
                                1."The group consists of two TEXT_OBJECT elements and one NUMERICDATA_OBJECT. To show a measure on screen, usually three elements are shown together: label, unit and value"  
                                {  
                                    "resolution_x": 1024,  
                                    "resolution_y": 768,  
                                    "window_info": {  
                                        "height": 600,  
                                        "width": 1024,  
                                        "object_infos": [  
                                            {  
                                                "height": 17,  
                                                "pos_x": 371,  
                                                "pos_y": 263,  
                                                "type": 21,  
                                                "type_name": "TEXT_OBJECT",  
                                                "width": 113,  
                                                "description":"label of a measure like temperature"  
                                            },  
                                            {  
                                                "height": 19,  
                                                "pos_x": 593,  
                                                "pos_y": 264,  
                                                "type": 21,  
                                                "type_name": "TEXT_OBJECT",  
                                                "width": 19,  
                                                "description":"unit label of a measure like °C"  
                                            },  
                                            {  
                                                "height": 18,  
                                                "pos_x": 527,  
                                                "pos_y": 264,  
                                                "type": 30,  
                                                "type_name": "NUMERICDATA_OBJECT",  
                                                "width": 64,  
                                                "description":"numeric input or display for a measure"  
                                            }  
                                        ]  
                                    },  
                                    "group_info": {  
                                        "group_width": 241,  
                                        "group_height": 20,  
                                        "group_pos_x": 371,  
                                        "group_pos_y": 263,  
                                        "description":"The group consists of two TEXT_OBJECT elements and one NUMERICDATA_OBJECT. To show a measure on screen, usually three elements are shown together: label, unit and value"  
                                    }  
                                }  
                                2. "The group is used to display a tank with scale, liquid level in 2D"
                                {
                                    "resolution_x": 0,
                                    "resolution_y": 0,
                                    "window_info": {
                                        "height": 768,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 310,
                                                "pos_x": 675,
                                                "pos_y": 279,
                                                "type": 22,
                                                "type_name": "GRAPH_OBJECT",
                                                "width": 196,
                                                "description":"A tank image"
                                            },
                                            {
                                                "height": 135,
                                                "pos_x": 793,
                                                "pos_y": 356,
                                                "type": 42,
                                                "type_name": "SHAPE_OBJECT",
                                                "width": 29,
                                                "description":"background of liquid level which indicates the inner-tank color, usually gray"
                                            },
                                            {
                                                "height": 133,
                                                "pos_x": 794,
                                                "pos_y": 357,
                                                "type": 37,
                                                "type_name": "BARGRAPH_OBJECT",
                                                "width": 27,
                                                "description":"a liquid level in the tank. A single bar to represents increasing or decreasing liquid level"
                                            },
                                            {
                                                "height": 134,
                                                "pos_x": 825,
                                                "pos_y": 356,
                                                "type": 83,
                                                "type_name": "DYNAMICSCALE_OBJECT",
                                                "width": 30,
                                                "description":"scale of liquid level (e.g. 0%~100%)"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 254,
                                        "group_height": 286,
                                        "group_pos_x": 690,
                                        "group_pos_y": 323,
                                        "description":"The group is used to display a tank with scale, liquid level in 2D"
                                    }
                                }
                                3. "The group consists of six numerics showing YEAR, MONTH, DAY, HOUR, MINUTE and SECOND. To make display more clear, using '/' separator to distinguish YEAR/MONTH/DAY. Units of HOUR, MINUTE and SECOND are also considered"
                                {
                                    "resolution_x": 0,
                                    "resolution_y": 0,
                                    "window_info": {
                                        "height": 768,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 39,
                                                "pos_x": 235,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for YEAR"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 140,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for MONTH"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 45,
                                                "pos_y": 373,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for DAY"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 45,
                                                "pos_y": 345,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 46,
                                                "description":"A text to describe DAY/MONTH/YEAR, usually called Date"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 130,
                                                "pos_y": 380,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 7,
                                                "description":"a text like '/' to separate DAY and MONTH"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 225,
                                                "pos_y": 380,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 7,
                                                "description":"a text like '/' to separate MONTH and YEAR"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 255,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for SECOND"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 139,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for MINUTE"
                                            },
                                            {
                                                "height": 39,
                                                "pos_x": 45,
                                                "pos_y": 453,
                                                "type": 29,
                                                "type_name": "NUMERICINPUT_OBJECT",
                                                "width": 79,
                                                "description":"numeric display for HOUR"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 45,
                                                "pos_y": 425,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 57,
                                                "description":"a text to describe HH:MM:SS, usually called Time"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 126,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 12,
                                                "description":"a text unit for HOUR, use 'h' to represent the unit"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 220,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 33,
                                                "description":"a text unit for MINUTE, use 'min' to represent the unit"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 335,
                                                "pos_y": 460,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 12,
                                                "description":"a text unit for SECOND, use 's' to represent the unit"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 302,
                                        "group_height": 147,
                                        "group_pos_x": 45,
                                        "group_pos_y": 345,
                                        "description":"The group consists of six numerics showing YEAR, MONTH, DAY, HOUR, MINUTE and SECOND. To make display more clear, using '/' separator to distinguish YEAR/MONTH/DAY. Units of HOUR, MINUTE and SECOND are also considered"
                                    }
                                }
                                4. "A guage or meter display to show a value from range 0-21000 kicks/hr. The group contains not only a meter, but also its numeric value and unit text"
                                {
                                    "resolution_x": 1920,
                                    "resolution_y": 1080,
                                    "window_info": {
                                        "height": 1080,
                                        "width": 1920,
                                        "object_infos": [
                                            {
                                                "height": 719,
                                                "pos_x": 627,
                                                "pos_y": 201,
                                                "type": 70,
                                                "type_name": "METERDISPLAY_CMT_OBJECT",
                                                "width": 719,
                                                "description":"A meter display with min=0 and max=21000"
                                            },
                                            {
                                                "height": 782,
                                                "pos_x": 597,
                                                "pos_y": 165,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 782,
                                                "description":"The bitmap acts as the background of meter. It can be also a monitor on a plc/hmi address"
                                            },
                                            {
                                                "height": 238,
                                                "pos_x": 696,
                                                "pos_y": 807,
                                                "type": 4,
                                                "type_name": "RECTANGLE_OBJECT",
                                                "width": 614,
                                                "description":"For purpose of beautification , a rectangle that wraps a numeric and a text to get user good understanding of meter value and its unit"
                                            },
                                            {
                                                "height": 140,
                                                "pos_x": 747,
                                                "pos_y": 821,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 514,
                                                "description":"numeric display wrapped in a rectangle for meter value"
                                            },
                                            {
                                                "height": 80,
                                                "pos_x": 879,
                                                "pos_y": 954,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 263,
                                                "description":"A text showing the unit of numeric display and wrapped in a rectangle"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 719,
                                        "group_height": 844,
                                        "group_pos_x": 627,
                                        "group_pos_y": 201,
                                        "group_name": "object_detail_MainScreen_group_5.png",
                                        "description":"A guage or meter display to show a value from range 0-21000 kicks/hr. The group contains not noly a meter, but also its numeric value and unit text"
                                    }
                                }
                                5. "The group represents a manual which contains two buttons (actually function keys) to achieve page switch."
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 69,
                                                "pos_x": 62,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the leftmost separator for function key"
                                            },
                                            {
                                                "height": 69,
                                                "pos_x": 162,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the middle separator to separate two function keys"
                                            },
                                            {
                                                "height": 69,
                                                "pos_x": 262,
                                                "pos_y": 513,
                                                "type": 2,
                                                "type_name": "LINE_OBJECT",
                                                "width": 1,
                                                "description":"The line object is the rightmost separator for function key"
                                            },
                                            {
                                                "height": 71,
                                                "pos_x": 80,
                                                "pos_y": 512,
                                                "type": 33,
                                                "type_name": "FUNKEY_OBJECT",
                                                "width": 71,
                                                "description":"The function key is used to page switch. In this case, page will be switched to 'home' as it is pressed"
                                            },
                                            {
                                                "height": 68,
                                                "pos_x": 176,
                                                "pos_y": 518,
                                                "type": 33,
                                                "type_name": "FUNKEY_OBJECT",
                                                "width": 74,
                                                "description":"The function key is used to page switch. In this case, page will be switched to 'Window 17' as it is pressed"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 201,
                                        "group_height": 74,
                                        "group_pos_x": 62,
                                        "group_pos_y": 512,
                                        "description":"The group represents a manual which contains two buttons (actually function keys) to achieve page switch."
                                    }
                                }
                                6. "The group consists of a bit lamp and a toggle switch. Here the toggle changes value of an address, and the bit lamp acts as a text object to describe its toggle"
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 69,
                                                "pos_x": 203,
                                                "pos_y": 249,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 195,
                                                "description":"This bit lamp is monitoring on an PLC address, and value of the address will affects state of the bimap. This bit lamp here acts as a text object to describe a toggle button as well"
                                            },
                                            {
                                                "height": 99,
                                                "pos_x": 84,
                                                "pos_y": 236,
                                                "type": 27,
                                                "type_name": "TOGGLESWITCH_OBJECT",
                                                "width": 99,
                                                "description":"The toggle switch (or called toggle button) can change value of an address, leading to the state change of a bit lamp"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 314,
                                        "group_height": 99,
                                        "group_pos_x": 84,
                                        "group_pos_y": 236,
                                        "description":"The group consists of a bit lamp and a toggle switch. Here the toggle changes value of an address, and the bit lamp acts as a text object to describe its toggle."
                                    }
                                }
                                7. "This group use a grid object as a table which size is 5 X 2. It completely contains current info of a running machine"
                                {
                                    "resolution_x": 1024,
                                    "resolution_y": 768,
                                    "window_info": {
                                        "height": 600,
                                        "width": 1024,
                                        "object_infos": [
                                            {
                                                "height": 211,
                                                "pos_x": 275,
                                                "pos_y": 380,
                                                "type": 85,
                                                "type_name": "GRID_OBJECT",
                                                "width": 570,
                                                "description":"The grid object acts as a table with 5 X 2 cells in this case. Then, user can drag any object into the cells where these objects are automatically arranged like an array"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 662,
                                                "pos_y": 512,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 80,
                                                "description":"The numeric object shows total batch weight of a machine. It's located at (4,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 328,
                                                "pos_y": 515,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 179,
                                                "description":"The text object describes batch weight of a machine. its content is 'Total Batch weight'. It's located at (4,1) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 348,
                                                "pos_y": 389,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 138,
                                                "description":"The text object describes state of a machine. its content is 'Batching State'. It's located at (1,1) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 582,
                                                "pos_y": 389,
                                                "type": 24,
                                                "type_name": "WORDLAMP_OBJECT",
                                                "width": 240,
                                                "description":"Unlike bit lamp (only ON/OFF), The word lamp object can show multi-state of a machine. It's located at (1,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 361,
                                                "pos_y": 557,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 113,
                                                "description":"The text object describes Fault Status of a machine. its content is 'Fault Status'. It's located at (5,1) of the grid object table"
                                            },
                                            {
                                                "height": 37,
                                                "pos_x": 665,
                                                "pos_y": 551,
                                                "type": 23,
                                                "type_name": "BITLAMP_OBJECT",
                                                "width": 74,
                                                "description":"The bit lamp shows Fault Status of a machine. It's located at (5,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 346,
                                                "pos_y": 431,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 142,
                                                "description":"The text object describes recipe of a machine. its content is 'Current Recipe'. It's located at (2,1) of the grid object table"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 573,
                                                "pos_y": 428,
                                                "type": 32,
                                                "type_name": "ASCIIDATA_OBJECT",
                                                "width": 259,
                                                "description":"The ascii object is aimed to show the current recipe value. It's located at (2,2) of the grid object table"
                                            },
                                            {
                                                "height": 25,
                                                "pos_x": 346,
                                                "pos_y": 473,
                                                "type": 21,
                                                "type_name": "TEXT_OBJECT",
                                                "width": 142,
                                                "description":"The text object describes weight of a machine. its content is 'Current Weight'. It's located at (3,1) of the grid object table"
                                            },
                                            {
                                                "height": 30,
                                                "pos_x": 662,
                                                "pos_y": 470,
                                                "type": 30,
                                                "type_name": "NUMERICDATA_OBJECT",
                                                "width": 80,
                                                "description":"The numeric object shows current weight of a machine. It's located at (3,2) of the grid object table"
                                            }
                                        ]
                                    },
                                    "group_info": {
                                        "group_width": 570,
                                        "group_height": 211,
                                        "group_pos_x": 275,
                                        "group_pos_y": 380,
                                        "description":"This group use a grid object as a table which size is 5 X 2. It completely contains current info of a running machine"
                                    }
                                }

                                In addition to the elements in the above, here is a list of permitted objects in EBPro for UI elements:
                                PIE_OBJECT, ELLIPSE_OBJECT, ARC_OBJECT, POLYGON_OBJECT, ANIMATION_OBJECT, VIDEO_OBJECT, ASCIIINPUT_OBJECT, QR_CODE_OBJECT, FILE_BROWSER_OBJECT,
                                PICTURE_VIEWER_OBJECT, ALARMDISPLAY_OBJECT, XYPLOT_OBJECT, GRID_OBJECT, SETWORD_OBJECT, COMBO_BUTTON_OBJECT, OPTIONLIST_OBJECT, WORDLAMP_OBJECT,
                                MULTISTATESWITCH_OBJECT, INDIRECTWINDOW_OBJECT, DIRECTWINDOW_OBJECT, TIMER_OBJECT, TRENDDISPLAY_OBJECT, RECIPEVIEW_OBJECT, PDF_READER_OBJECT,
                                LINE_OBJECT, SLIDER_OBJECT


                                please generate another json file in the above format, that represents a UI with the following rules and description:
                                rule1: You should fully define the layout pattern and cannot simplify your result.
                                rule2: Do not explain any words and add any annotation in your json file.
                                rule3: The type of object you generate must be one of the list of permitted objects and examples
                                rule4: Try to generate at least 3-word description of each element accroding to user query and based on its functional, color, shape ... etc. in English.
                                rule5: 
                                Return your result in the following format:
                                ```
                                <put your result here>
                                ```
                                
                                The description:
                            """
        
    def generate_layout_json(self, query):
        prompt = self.fewShotPrompt + query
        replied_message = ''
        max_try = 3
        try_cnt = 0
        while try_cnt < max_try:
            try:
                resp = AWSFunction.chat_completion_anthropic(prompt=prompt)
                status = resp["status"]
                if status == 'success':
                    replied_message = resp['replied_message']
                    # print(replied_message)
                    break
                else:
                    print(resp["error_reason"])
                    try_cnt += 1
            except Exception as e:
                print(str(e))
                try_cnt += 1
        return prompt, replied_message
    
    def optimize_layout(self, history_messages , query):
        llm_history_messages = AWSFunction.convert2HistoryMessages(history_messages=history_messages, 
                                                                    query=query)
        replied_message = ''
        max_try = 3
        try_cnt = 0
        while try_cnt < max_try:
            try:
                resp = AWSFunction.chat_completion_anthropic_history(llm_history_messages)
                status = resp["status"]
                if status == 'success':
                    replied_message = resp['replied_message']
                    # print(replied_message)
                    break
                else:
                    print(resp["error_reason"])
                    try_cnt += 1
            except Exception as e:
                print(str(e))
                try_cnt += 1
        return query, replied_message
    
    
    
    
    

if __name__ == '__main__':
    layout_gpt = FewShotLearningGPT()
    query = """
            我想要一張圖顯示barcode，一個文字在barcode上方顯示contact us 在barcode正右方置入一個XY Plot，在XY Plot下方置入一個pie chart。
            barcode、XY Plot、Pie Chart大小同大。頁面最左方放入五個垂直排列的按鈕進行頁面切換
            """
    t_start = time.time()
    prompt, generation_text = layout_gpt.generate_layout_json(query)
    print(generation_text)
    json_str_data, json_data = layout_gpt.catch_json_from_generation(generation_text)
    print('\n',json_data)
    img_base64  = layout_gpt.plot_data(json_data,True)
    # print(img_base64)
    t_end = time.time()
    print(f'Time cost: {t_end - t_start} s')
    pass