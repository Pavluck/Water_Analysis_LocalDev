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
          <container classname="Form">
    <form onSubmit={handleSubmit} id="water-form" ref={formRef}>
      <p style={{fontSize: '20px', padding:'2px', margin:'5px'}}>PH determines if the component is acidic, neutral, or alkaline.</p>
      <label style={{fontWeight:'900px'}}>
        PH Level:  
        <input style={{ width:'250px', height: '40px', fontSize: '25px', padding:'2px', margin:'5px'}} type="number" min="0" max="14" step="0.1" placeholder="0 - 14" value={ph_level} onChange={(e) => setph_level(e.target.value)} />
      </label>
      <p>Low levels of Chlorine are safe and kill bacteria.<br/> It is measured in miligrams per Liter</p>
      <label style={{fontWeight:'900px'}}>
        Chlorine:
        <input style={{ width:'250px', height: '40px', fontSize: '25px', padding:'2px', margin:'5px'}} type="number" min="0" max="100000000" step="0.5" placeholder="0 - 1,000,000" value={chlorine} onChange={(e) => setchlorine(e.target.value)} />
      </label>
        <p>The color of the water can determine certain attributes. This is an optional parameter. <br/>If you are able to, you may select a color that best matches the body of water.
      </p>
      <label style={{fontWeight:'900px'}}>Color:
        <div className="Color">
          <CirclePicker color={watercolor} onChangeComplete={handleWaterColorChange} />
        </div>
      </label>
      <p>
        Total Dissolved Solids (TDS) occur naturally from minerals by rocks and soil, runoff from rivers and streams.<br/> However, too much affects the potability of water. <br/>
      </p>
      <p><label style={{fontWeight:'900px'}}>
        Enter the amount of TDS in mg per L:
        <input style={{ width:'250px', height: '40px', fontSize: '25px', padding:'2px', margin:'5px'}} type="number" min="0" max="100000000" step="0.1" placeholder="0 - 1,000,000" value={TDS} onChange={(e) => setTDS(e.target.value)} />
      </label>
      </p>
      <p>
        High bacteria levels can also pose a health risk. <br/>Bacteria levels are measured in terms of colony-forming units (CFU), <br/>which is the number of individual bacterial cells that can grow into a visible colony on a plate. 
      </p>
      <label style={{fontWeight:'900px'}}>
        Enter the amount of colony-forming units (CFU) in miligrams per Liter:
        <input style={{ width:'250px', height: '40px', fontSize: '25px', padding:'2px', margin:'5px'}} type="number" min="0" max="100000000" step="0.1" placeholder="0 - 1,000,000" value={CFU} onChange={(e) => setCFU(e.target.value)} />
      </label><br/>
      <p>
        Algae typically appears in polluted water. <br/>Algal blooms make the water body appear green
      </p>
      <label style={{fontWeight:'900px'}}>
        Is there algae present in this body of water?
        <input style={{ width:'45px', height: '30px', padding:'2px', margin:'2px'}} type="checkbox" checked={algae} onChange={(e) => setalgae(!algae)} />
      </label>
    </form></container>
    </div>
  )
}
export default WaterAIform;

