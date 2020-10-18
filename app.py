from flask import Flask,jsonify,request
from flask_cors import CORS
import kickstarterData

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/kickstarter')
def kickstarter():
    print("request.args", request.args)
    if(request.args == None):
        return "Invalid"
    result = kickstarterData.process(request.args)
    print("result", result)
    response = jsonify({"result": result["prediction"][0][1], "error": None})
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

if __name__ == '__main__':
    app.run()
