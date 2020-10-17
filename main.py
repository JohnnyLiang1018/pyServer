from flask import Flask,jsonify,request
from flask_cors import CORS
import kickstarterData

app = Flask(__name__)
CORS(app)

@app.route('/kickstarter')
def kickstarter():
    print("request.args", request.args)
    result = kickstarterData.process(request.args)
    print("result", result)
    return jsonify({"result": result["prediction"][0][1], "error": None})

if __name__ == '__main__':
    app.run()
