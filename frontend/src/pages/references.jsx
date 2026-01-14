/* ------------ References Page for the credits/refernces of libraries used by the Water software ------------*/

/* ------------ Necessary Imports ------------*/
import React from 'react'
import { Link } from "react-router-dom"
import { CgDrop } from "react-icons/cg";
import { useNavigate } from 'react-router-dom';
import Home from './home';
/* ------------ Navigation ------------*/

function References() {
    const navigate = useNavigate();
  
    const Aboutbutton = () => {
      navigate('/home');
    };
  return (
    /* ------------ Page Content  ------------*/

    <div className="App">
      <header className="App-header">
      <h1><CgDrop></CgDrop> Water Analysis AI <CgDrop></CgDrop></h1></header>
      <nav>
      <button className="waterbutton" onClick={Aboutbutton}>Home</button> 
      </nav>
        <h2> About the Water AI </h2>
            <p>The Water Analysis is created using a Modified Bayes Decision Tree for Supervised Machine Learning</p>
            Provides data anaysis based on:<br/>
        <ul style={{ fontSize: '25px', padding:'15px', margin:'5px', width:'500px'}}>
        <li>PH level</li> 
        <li>Chlorine Concentration</li>
        <li>Color</li>
        <li>Total dissolved solids (TDS)</li>
        <li>Bacteria levels</li>
        <li>Presence of algae</li>
      </ul>
        <h2 >Libraries and references used are:</h2>
        <ul>
        <li></li> 
        <a href="https://numpy.org/">Numpy</a>
        <p></p>
        <li></li>
        <a href="https://scikit-learn.org/stable/">SciKit-Learn</a>
        <p></p>
        <li></li>
        <a href="https://www.researchgate.net/publication/365495813_Water_Potability_Analysis_and_Prediction">Studies on Water Potability</a>
        <p></p>
        <li></li>
        <a href="https://react-icons.github.io/react-icons/">React Icons</a>
      </ul>
    </div> 
  )
}
export default References
/* ------------ Natasha Pavelek - 2026------------*/
