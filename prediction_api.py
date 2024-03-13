"""
Date: 3/13/24
Author: Tayven Stover

This API is designed to return the predictions of a machine learning algorithm.

Example Response Format:
{
    datetime: 2022-05-17 03:00:00,
    raw_data: pm25_cf_1,
    predicted_data: prediction
}
"""
import flask
import predict

app = flask.Flask(__name__)
@app.route('/', methods=['GET'])
def get_predictions():
    prediction_results = predict.format_results()
    return prediction_results

if __name__ == "__main__":
    app.run(port = 5000)