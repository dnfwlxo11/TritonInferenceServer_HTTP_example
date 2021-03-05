import numpy as np
import time
import json
import requests
import os
import cv2

def inference(image, host='localhost', port=8000):
    try:
        image = np.expand_dims(image, 0)
        image = image.astype(np.uint8)
        batch, row, col, ch = image.shape
        image = image.tobytes()
        
        data = image

        body = {
            "inputs":[
                {
                    "name":"image_arrays:0",
                    "shape":[batch,row,col,ch],
                    "datatype":"UINT8",
                    "parameters":{
                        "binary_data_size":len(data)
                    }
                }
            ],
            
            "parameters":{
                "binary_data_output": True
            }
        }
      
        body = json.dumps(body).replace(' ', '')
        body = body.encode()
        data = body + data

        infer_content_length = len(body)

        header = {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(infer_content_length),
            "Accept": "*/*"
        }
    
        start_time = time.time()
        model_name = 'mymodel'
        response = requests.post('http://' + host + ':' + str(port) + '/v2/models/' + model_name + '/versions/1/infer', 
                                data=data, headers=header)
        print("* Inference time: {:.2f}".format(time.time() - start_time))

        results = response.content[infer_content_length+2:]
        results = np.frombuffer(results, np.float32).reshape((-1, 7))

        print('output shape:', results.shape)
        prediction = results

        boxes = prediction[:, 1:5]
        classes = [label[i - 1] for i in prediction[:, 6].astype(int)]
        scores = prediction[:, 5]

        for i in range(len(scores)):
            if scores[i] > 0.65:
                print('*'*75)
                print('box pos: {}\nclass: {}\nscore: {}'.format(boxes[i], classes[i], scores[i]))
                print('*'*75)
            else:
                break

    except Exception as e:
        print(e)

img = cv2.imread('./person.jpg')

inference(img)
