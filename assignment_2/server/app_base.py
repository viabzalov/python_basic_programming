from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/www')
def hello_world1():
    return 'Hello World1!'


if __name__ == '__main__':
    app.run()
