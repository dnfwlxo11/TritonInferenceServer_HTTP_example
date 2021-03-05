import argparse
import numpy as np
import sys
import cv2

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

def infer(model_name, _input, input_li, headers=None):
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('image_arrays:0', input_li, "UINT8"))

    inputs[0].set_data_from_numpy(_input, binary_data=True)

    results = triton_client.infer(model_name,
                                  inputs,
                                  outputs=outputs,
                                  headers=headers)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8000',
                        help='Inference server URL. Default is localhost:8000.')
    parser.add_argument(
        '-H',
        dest='http_headers',
        metavar="HTTP_HEADER",
        required=False,
        action='append',
        help='HTTP headers to add to inference server requests. ' +
        'Format is -H"Header:Value".')

    FLAGS = parser.parse_args()
    try:
        triton_client = httpclient.InferenceServerClient('localhost:8000', verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    model_name = "mymodel"

    img = cv2.imread('./person.jpg')
    input_data = np.expand_dims(img, axis=0)
    input_data = input_data.astype(np.uint8)
    batch, row, col, ch = input_data.shape

    if FLAGS.http_headers is not None:
        headers_dict = {
            l.split(':')[0]: l.split(':')[1] for l in FLAGS.http_headers
        }
    else:
        headers_dict = None

    results = infer(model_name, input_data, [batch, row, col, ch], headers_dict)

    # statistics = triton_client.get_inference_statistics(model_name=model_name, headers=headers_dict)
    # print(statistics)
    # if len(statistics['model_stats']) != 1:
    #     print("FAILED: Inference Statistics")
    #     sys.exit(1)

    label = ['test', 'test2']
    results = results.as_numpy('detections:0')
    print('output shape:', results.shape)
    prediction = np.frombuffer(results, np.float32).reshape((-1, 7))

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

    sys.exit(1)
