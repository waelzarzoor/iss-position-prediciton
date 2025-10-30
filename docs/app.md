### - app.py

The `app.py` file implements a Flask web application for predicting the position of the International Space Station (ISS) in real-time using a trained model. Below is a breakdown of the components and instructions for running the application:

#### Functionality

- **Flask Application Setup**: Initializes a Flask application.
- **GetData Class**: Defines a class for fetching real-time data and making predictions using a trained model.
- **Flask Routes**:
  - `/`: Renders the main HTML template.
  - `/stream`: Streams real-time ISS data and predictions to the client.
- **HTML Template**: Provides a simple HTML page with a Plotly chart to visualize the true and predicted positions of the ISS.
- **JavaScript**:
  - Sets up a Plotly chart for real-time visualization.
  - Establishes an EventSource to stream data from the server.
  - Updates the Plotly chart with new data received from the server.

#### Usage

1. Ensure you have Python and the required dependencies installed.
2. **Navigate to the `is-position-prediction` directory** in your terminal.
3. Run the following command in your terminal to start the Flask application:

    ```bash
    python src/scripts/app.py
    ```

4. The application will start, and the address where you can access it will be displayed in the terminal, typically in the format `http://127.0.0.1:3000`.

5. Open your web browser and navigate to the displayed address to access the ISS prediction web interface.

Note: By default, the script utilizes a pre-trained model. However, users can choose to use a user-trained model by specifying the `--user` flag. This flag modifies the behavior of the script to load the model checkpoint path suitable for user-trained model.
When utilizing Docker for running applications, it is recommended to launch applications through Docker Desktop rather than directly from links in the terminal. 