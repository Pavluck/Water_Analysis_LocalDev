/* ------------ Page to allow the user to give data about a body of water ------------*/

/* ------------ Necessary Imports ------------*/
import React, { useState, useRef} from 'react';
import { CirclePicker } from 'react-color';
import axios from 'axios';
import { CgDrop } from "react-icons/cg";
import { useNavigate } from 'react-router-dom';
import References from './references';
import Home from './home';

/* ------------ Navigation ------------*/
function WaterAIform() {
  const navigate = useNavigate();
  const handleClick = () => {
    navigate('/home');
  };
  const Aboutbutton = () => {
    navigate('/references');
  };
/* ------------ Page Content: Form for UI ------------*/
  return (
    <div>
      <header className="Form"><h1>Water Form</h1></header>
      <nav>
        <button className="waterbutton" onClick={handleClick}>Home</button> <button className="waterbutton" onClick={Aboutbutton}>About</button>
      </nav>
      <h2>Data Form</h2>
        <p>
        A water potability test is an assessment that determines if water is safe for human consumption <br/>
        It checks for harmful contaminants, such as bacteria, algae and chemicals.<br/>
        </p>
      <fieldset>
        <legend> Please enter some information about the body of water to be analyzed</legend>      
      </fieldset>
    </div>
  )
}
export default WaterAIform;

