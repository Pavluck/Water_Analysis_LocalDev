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
from water_analysis import water_analysis
# ~~~~~ Port ~~~~
app = Flask(__name__,
            static_url_path = '',
            static_folder='static',)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024*1024
app.register_blueprint(water_analysis)
# Error Handling 
@app.errorhandler(413)
def err_size(_error):
            return jsonify({'error': 'please select a file 100 MB or less.'}), 413
# ~~~~ CORS ~~~~
CORS_ORIGINS = os.getenv('CORS_ORIGINS')
CORS(app,
     origins=os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(','),
     supports_credentials=True)

# ~~~~~~~ Functions & Formulas ~~~~~~~
@app.route('/decision', methods=['POST'])
def decision():
    """
    AI not needed for absolutes.
    This uses data to decide if water is potable or not - sends decision to the frontend
    """
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
            if 'color' in data and data['color'] is not None and data['color'] != '':
                        color = data['color']
                        color_hsv = color['hsv']
                        hue = float(color_hsv['h'])
                        # analysis water that is too vivid
                        if 174.0 < hue < 232:
                                    sat = float(color_hsv['s'])
                                    if 0.0 <= sat < 0.92:
                                                pass
                                    else:
                                                decision = "This water's coloration is oddly vivid but may be potable"
            decision_json = {
                    "decision": decision,
                }

            # return code 200 for the frontend if success    
            return jsonify(decision_json), 200
# ~~~~~~~~~~ Routes & Returns ~~~~~~~~~~
@app.route('/')
@app.route('/index.html')
def index():
    print("Index accessed")
    return app.send_static_file('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', 'SECRET')) # 1235
# EOF
