#!flask/bin/python
from app import app
import argparse

parser = argparse.ArgumentParser(prog='Mapper Interactive')
parser.add_argument("-p", "--port", default=8080)

args = parser.parse_args()

app.run(host='127.0.0.1',port=args.port,debug=True)
