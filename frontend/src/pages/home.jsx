/* ------------ Home page for the water software ------------*/

/* ------------ Necessary Imports ------------*/
import React from 'react';
import { Link } from "react-router-dom"
import { useNavigate } from 'react-router-dom';
/* ------------ TODO add Navigation ------------*/

/* ------------ Page Content  ------------*/
const Home = () => {
  return (
    <div className="App">
      <h1>Water Analaysis</h1>
        <p style={{ fontSize: '29px', padding:'2px', margin:'5px'}}>
        Welcome!
        This software will determine whether a body of water is potable. </p><br />
        <p style={{ fontSize: '25px', padding:'2px', margin:'5px'}} >Given an input of data, the AI will analysis characteristics
        to answer if a body of water is potable for humans.
        A water potability test is an assessment that determines if water is safe for human consumption 
        by analyzing its physical, chemical, and microbiological properties.
        It checks for harmful contaminants, such as bacteria, algae and chemicals.
        <br /><br /></p>
    </div>
  );
};

export default Home;
/* ------------ Natasha Pavelek - 2026------------*/
