# Water Analysis: Local Development
This is different than the deployed version. To see my deployed Water Analysis, visit here: https://water-analysis.onrender.com/  
Note: React takes a *very* long time to spin up on Render, taking about 8 minutes for both backend and frontend to be responsive.  

<img width="784" height="1134" alt="Screenshot 2026-01-04 195932" src="https://github.com/user-attachments/assets/8a758e42-6fa7-4132-a800-c87d9ad1b6b8" />

The Water Analysis software determines whether a body of water is potable for humans.
Uses Chemical and/or Physical Features from the frontend (a website) to send the data to the backend.
The backend analyzes the input, and returns a decision to the frontend for a user.

# Project Setup and Run Instructions  
This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app)  

## Requirements  
This project requires Python with FLASK along with Flask-CORS for security measures   
Instruction on how to [install Python](https://wiki.python.org/moin/BeginnersGuide/Download)  

## Configure for Python using Visual Studio Code  
1. Make sure you have the Python extension in VS Code (Visual Studio Code)
2. Make sure Python is installed on your Operating System
3. In VS Code, Hit CTRL+Shift+P 
4. When the drop down appears, select "Python: Create Enviroment"
5. When prompted, select "Venv" 
6. When prompted, select your latest Python version that is installed
7. Wait until configuration is complete
8. Upon completion, you can use the Python virtual enviroment by selecting at the top menu bar: "..." --> "Terminal" --> "New Terminal"
9. Install the necessary Python packages and dependencies described in the Dependencies section below

## Project Structure
frontend/  
└── Water Analysis Application # React Web App  
│ ├── App.js # React main page for routing  
| └── components/ # React components / react-router-dom  
| └── public # HTML
backend/  
└── Functions to train and feed user input to the AI (under development)  
