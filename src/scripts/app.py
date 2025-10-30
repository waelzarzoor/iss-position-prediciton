from flask import Flask, render_template, Response
import plotly.graph_objects as go
import json
import torch
import argparse
from src.modules.iss_data_fetcher import FetchData
from src.modules.predictor_module import LightningLatLongPredictor
from src.modules.utils import get_model_checkpoint_path

app = Flask(__name__)

class GetData():
    def __init__(self, model):
        self.fetcher = FetchData()
        if model == 'user':
            model_checkpoint = get_model_checkpoint_path(model='user')
        else:
            model_checkpoint = get_model_checkpoint_path(model='pretrained')
        print(model_checkpoint)
        self.model = LightningLatLongPredictor.load_from_checkpoint(
            checkpoint_path=model_checkpoint[0],
            hparams_file=model_checkpoint[1],
            map_location=torch.device('cpu'))
    
    def data_and_preds(self):
        while True:
            net_input_longitude = []
            net_input_latitude = []

            data = list(map(float, next(self.fetcher)))
            net_input_longitude.append(data[0])
            net_input_latitude.append(data[1])

            net_input = torch.tensor([net_input_longitude, net_input_latitude], dtype=torch.float32)
            net_input = net_input.T
            net_input = net_input.reshape(1, 1, 2)
            
            net_input = net_input / 180
            net_output = self.model(net_input)
            pred = net_output * 180
            pred = torch.reshape(pred, (2, 1))

            longitude = net_input_longitude[0]
            latitude = net_input_latitude[0]
            pred_longitude = pred[0]
            pred_latitude = pred[1]

            yield {
                'longitude': float(longitude),
                'latitude': float(latitude),
                'pred_longitude': round(float(pred_longitude), 4),
                'pred_latitude': round(float(pred_latitude), 4)
            }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    get_data = GetData(app.config['MODEL'])
    def generate():
        for data_point in get_data.data_and_preds():
            yield f"data: {json.dumps(data_point)}\n\n"
    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', action='store_true', help='Use user-defined model')
    args = parser.parse_args()

    if args.user:
        app.config['MODEL'] = 'user'
    else:
        app.config['MODEL'] = 'pretrained'

    app.run(host='0.0.0.0', port=3000, threaded=True)