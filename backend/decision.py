"""
NP: Frontend to Backend Communication
Accepts data from the frontend and communicates with the backend to return a decision to the frontend
"""
# ~~~~~~~~~~ Necessary Imports ~~~~~~~~~~
import os
from random import randint
import json
from flask import *
from flask_cors import CORS
# ~~~~~ Port ~~~~
app = Flask(__name__,
            static_url_path = '',
            static_folder='static',)
# ~~~~ CORS ~~~~
CORS(app,
     origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','),
     supports_credentials=True)

# ~~~~~~~ Functions & Formulas ~~~~~~~
@app.route('/decision', methods=['POST'])
def decision():
            """Uses the data from the frontend and sends the decision whether the 
            water is potable or not to the frontend"""
            data = request.get_json()
            if not data:
                        return jsonify({'error': 'No input data provided'}), 400
            print(data)
            #default safeguard 
            decision = "Oops, water not found! Analysis incomplete")
            ph_level = None
            chlorine = None
            color = None
            TDS = None
            CFU = None
            algae = None
            # TODO: validate data from frontend

