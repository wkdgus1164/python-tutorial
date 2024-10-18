from flask import Flask, request, jsonify
from holoviews.operation import method

from sample import calculate

app = Flask(__name__)


@app.route("/", methods=['POST'])
def hello_world():
    query = request.args.get('query', type=str)
    print(query)
    print(type(query))
    result = calculate(query)
    return jsonify(result)


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
