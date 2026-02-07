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
