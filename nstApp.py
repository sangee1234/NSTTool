from flask import Flask, request
from NST import NST
import concurrent.futures
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def testMethod():
    return "Working!!"

@cross_origin()
@app.route('/combine', methods=['POST', 'OPTION'])
def combineImages():
    print(request.files)
    # style_file = request.files['style_image'].stream
    # content_file = request.files['content_image'].stream
    # with concurrent.futures.ThreadPoolExecutor as executer:
    #     futures = executer.submit(runNST, style_file, content_file)
    #     return futures.result
    return {"data":"1"}

def runNST(style_image, content_image):
    nst = NST(style_image, content_image)
    return nst.train()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)

