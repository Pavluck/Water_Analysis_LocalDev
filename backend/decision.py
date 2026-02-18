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
            
            # validate data from frontend
            if 'ph_level' in data and data['ph_level'] is not None and data ['ph_level'] != '':
                        try:
                                    ph_level = float(data['ph_level'])
                        except:
                                    return jsonify({'error: PH level is invalid, needs to be a value between 0 & 14'},400)   
            if 'chlorine' in data and data['chlorine'] is not None and data ['chlorine'] != '':
                        try:
                               chlorine = float(data['chlorine'])
                        except:
                                    return jsonify({'error: Chlorine is invalid, needs to be a value between 0 and 1 million'}, 400)            
            # TODO: validate for other features
