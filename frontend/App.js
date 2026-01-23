// ------- Natasha Pavelek 2026 -----------

// ------- Necessary Imports -----------
import './tealwater.css';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Home from "./pages/home";
import WaterAIform from "./pages/dataform";
import References from "./pages/references";

// ~~~~~~ Page Content ~~~~~~
function App() {
  return (
      <div className='App'>
        <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/home" element={<Home />} />
          <Route path="/dataform" element={<WaterAIform />} />
          <Route path="/references" element={<References />} />
        </Routes>
        </Router>
      <footer>
            <p> &copy; 2026 Natasha Pavelek</p>
      </footer>
      </div>
  );
}
export default App;
