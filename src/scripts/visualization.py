import torch
import argparse
from src.modules.iss_data_fetcher import FetchData
from src.modules.predictor_module import LightningLatLongPredictor
from src.modules.utils import draw_earth, draw_points, get_model_checkpoint_path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torchmetrics import MeanAbsoluteError


animation_paused = False

def pause_animation(event):
    global animation_paused
    animation_paused = not animation_paused

def update_plot(frame, model, ax, lon_lat, preds):
    global animation_paused
    
    ax.cla()
    draw_earth(ax)

    fetch_data = FetchData()

    net_input_longitude = []
    net_input_latitude = []
    true_lon_lat = []

    for i in range(2):

        if i < 1:
            data = list(map(float, next(fetch_data)))
            net_input_longitude.append(data[0])
            net_input_latitude.append(data[1])
        else:
            lon, lat = list(map(float, next(fetch_data)))
            true_lon_lat.append([lon, lat])

    net_input = torch.tensor([net_input_longitude, net_input_latitude], dtype=torch.float32)
    net_input = net_input.T
    net_input = net_input.reshape(1, 1, 2)

    true_lon_lat = torch.tensor(true_lon_lat, dtype=torch.float32)

    net_input = net_input / 180
    net_output = model(net_input)
    pred = net_output * 180

    pred = torch.reshape(pred, (2, 1))
    true_lon_lat = torch.reshape(true_lon_lat, (2, 1))

    preds.append(pred)
    lon_lat.append(true_lon_lat)

    mae = MeanAbsoluteError()
    mean_abs_err = mae(pred, true_lon_lat)

    draw_points(ax, lon_lat, label='True Position')
    draw_points(ax, preds, label='Predicted Position')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    text_true_lon = f'True Longitude: {round(lon_lat[-1][0].item(), 3)}'
    text_true_lat = f'True Latitude: {round(lon_lat[-1][1].item(), 3)}'
    text_pred_lon = f'Pred Longitude: {round(preds[-1][0].item(), 3)}'
    text_pred_lat = f'Pred Latitude: {round(preds[-1][1].item(), 3)}'
    text_mae = f'MAE: {round(mean_abs_err.item(), 3)}'

    text = f'{text_true_lon}\n{text_pred_lon}\n{text_true_lat}\n{text_pred_lat}\n{text_mae}'
    ax.text2D(0.95, 0.0, text, transform=ax.transAxes, fontsize='x-small')

    if animation_paused:
        ani.event_source.stop()
    else:
        ani.event_source.start()

    ax.legend()
    ax.set_title('ISS Position Prediction')

def main(model):
    global ani
    print(model)
    if model == True:
        model_checkpoint = get_model_checkpoint_path(model='user')
    else:
        model_checkpoint = get_model_checkpoint_path(model='pretrained')

    lit_model = LightningLatLongPredictor.load_from_checkpoint(
        checkpoint_path=model_checkpoint[0],
        hparams_file=model_checkpoint[1],
        map_location=torch.device('cpu'))
    
    lit_model.eval()

    lon_lat = []
    preds = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ani = FuncAnimation(fig, update_plot, fargs=(lit_model, ax, lon_lat, preds), frames=100, interval=100)

    fig.canvas.mpl_connect('key_press_event', pause_animation)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', action='store_true', help='Use user-defined model')
    args = parser.parse_args()

    main(model=args.user)