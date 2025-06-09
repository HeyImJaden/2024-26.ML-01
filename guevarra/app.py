from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    features = data.get("features")

    my_model = joblib.load(".../linear_model.joblib")
    infer_result = my_model.predict(features)

    response_data = {
        "result": {
            "value": infer_result
        }
    }
    
    return jsonify(response_data)

@app.route('/infer', methods=['GET'])
def hello2():
    # data = request.get_json()
    # name = data.get('name', 'Stranger')
    return "<h1>ciao</h1>"

if __name__ == '__main__':
    app.run(debug=True)