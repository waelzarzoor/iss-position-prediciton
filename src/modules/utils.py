import os
import numpy as np
import matplotlib.pyplot as plt

EARTH_RADIUS = 6371
ISS_ORBIT_RADIUS = 420

def polar_to_cartesian(lon, lat, radius):
    '''
    Convert polar coordinates (longitude, latitude) to Cartesian coordinates (x, y, z).

    Parameters:
    - lon (float): Longitude in degrees.
    - lat (float): Latitude in degrees.
    - radius (float): The radius from the origin to the point in space.

    Returns:
    Tuple[float, float, float]: Cartesian coordinates (x, y, z).

    The function takes longitude, latitude, and radius as input and converts
    them to Cartesian coordinates. The resulting coordinates represent a point
    in 3D space on the surface of a sphere centered at the origin.

    Example:
    x, y, z = polar_to_cartesian(45.0, 30.0, 10.0)
    '''

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    return x, y, z

def draw_earth(ax, earth_radius=EARTH_RADIUS):
    '''
    Draw a 3D representation of the Earth on the specified Axes.

    Parameters:
    - ax (matplotlib.axes._subplots.Axes3D): The 3D Axes on which to draw on.

    Description:
    This function generates a 3D plot of the Earth using a spherical coordinate system.
    It uses the provided Axes object to draw the Earth's surface as a solid sphere.
    The surface color is set to green with some transparency (alpha=0.2), and the linewidth is set to 0.

    Note:
    This function assumes the global variable EARTH_RADIUS is defined and represents the radius of the Earth.

    Example:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_earth(ax)
    plt.show()
    '''

    phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]
    x = earth_radius * np.sin(theta)*np.cos(phi)
    y = earth_radius * np.sin(theta)*np.sin(phi)
    z = earth_radius * np.cos(theta)
    ax.plot_surface(x, y, z, color='green', alpha=0.2, linewidth=0)

def draw_points(ax, data, label, earth_radius=EARTH_RADIUS, iss_orbit_radius=ISS_ORBIT_RADIUS):
    '''
    Scatter plot points on a 3D Axes.

    Parameters:
    - ax (matplotlib.axes._subplots.Axes3D): The 3D Axes to draw on.
    - data (list): List of 2D tensors containing longitude and latitude coordinates.
    - label (str): Label for the points, should be either 'Pred' or 'True'.
    - earth_radius (float, optional): Earth's radius. Default is the global variable EARTH_RADIUS.
    - iss_orbit_radius (float, optional): Radius of the ISS orbit. Default is the global variable ISS_ORBIT_RADIUS.

    The function takes 3D coordinates (longitude, latitude) from the input data
    and converts them to Cartesian coordinates. It then scatter plots these
    points on the provided 3D Axes with different colors based on the label.
    The last point is marked with a cross.

    Example:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = torch.tensor([[45.0, 30.0], [60.0, 40.0], [70.0, 50.0]])
    draw_points(ax, data, label='True')
    plt.show()
    '''

    if label == 'Predicted Position':
        color = 'red'
        color_last = 'orange'
    elif label == 'True Position':
        color = 'blue'
        color_last = 'cyan'
    else:
        raise ValueError(f'Pick one from ["Predicted Position", "True Position"].')

    for i in range(len(data)):

        longitude, latitude = data[i].detach().numpy()
        x, y, z = polar_to_cartesian(longitude, latitude, earth_radius + iss_orbit_radius)

        if i == 0:
            ax.scatter(x, y, z, c=color, marker='o', label=label)
        elif i < (len(data)-1):
            ax.scatter(x, y, z, c=color, marker='o')
        else:
            ax.scatter(x, y, z, c=color_last, marker='x')
    
class FolderNotFoundError(Exception):
    pass

def get_model_checkpoint_path(model: str):
    '''
     Retrieves the paths to the checkpoint file and the hparams.yaml file for the specified model.

    Args:
    - model (str): A string indicating the type of model to retrieve the paths for.
      Possible values:
      - 'pretrained': Indicates that the paths for a pretrained model should be retrieved.
      - 'user': Indicates that the paths for a user model should be retrieved.

    Returns:
    - If 'model' is 'pretrained':
      - pretrained_model_checkpoint_path (str): The path to the pretrained model checkpoint file.
      - pretrained_model_hparams_path (str): The path to the hparams.yaml file for the pretrained model.

    - If 'model' is 'user':
      - ckpt_file_path (str): The path to the checkpoint file for the user model.
      - hparams_yaml_path (str): The path to the hparams.yaml file for the user model.

    Raises:
    - FolderNotFoundError: If the 'lightning_logs' folder is not found in the current directory.
    - FolderNotFoundError: If no version folders are found within the 'lightning_logs' directory.
    - FileNotFoundError: If no checkpoint file is found within the 'checkpoints' folder for the user model.
    - ValueError: If an invalid value is provided for the 'model' parameter.
    '''

    pretrained_model_checkpoint_path = 'src/checkpoints/pretrained_model.ckpt'
    pretrained_model_hparams_path = 'src/checkpoints/hparams.yaml'
    
    if model == 'pretrained':
        return pretrained_model_checkpoint_path, pretrained_model_hparams_path
    
    elif model == 'user':

        current_directory = os.getcwd()
        lightning_logs_path = os.path.join(current_directory, 'lightning_logs')

        if not os.path.exists(lightning_logs_path):
            raise FolderNotFoundError("Folder 'lightning_logs' not found. Please execute training using the 'train_pipeline.py' script.")

        version_folders = [folder for folder in os.listdir(lightning_logs_path)]
        if not version_folders:
            raise FolderNotFoundError("No version folders found in 'lightning_logs' directory.")

        lastest_version_folder = version_folders[0]

        latest_version_folder_path = os.path.join(lightning_logs_path, lastest_version_folder)

        hparams_yaml_path = os.path.join(lightning_logs_path, latest_version_folder_path, 'hparams.yaml')

        checkpoints_folder = os.path.join(lightning_logs_path, latest_version_folder_path, 'checkpoints')
        checkpoints = os.listdir(checkpoints_folder)

        ckpt_files = [file for file in checkpoints if file.endswith('.ckpt')]
        if not ckpt_files:
            raise FileNotFoundError("No checkpoint file found in the 'checkpoints' folder.")
        
        ckpt_file = ckpt_files[0]
        ckpt_file_path = os.path.join(checkpoints_folder, ckpt_file)

        return ckpt_file_path, hparams_yaml_path
    
    else:
        raise ValueError('Invalid value for "model". Use "pretrained" or "user".')