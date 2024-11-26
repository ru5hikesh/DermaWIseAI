//import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <ul>
          <li><Link to="/Home">Home</Link></li>
          <li><Link to="/About">About</Link></li>
          <li><Link to="/Login">Login</Link></li>
          <li><Link to="/Register">Register</Link></li>
          <li><Link to="/Service">Services</Link></li>
          <li><Link to="/Contact">Contact</Link></li>
        </ul>
        
        <div className="profile-section">
          <span className="profile-name">SKIN DISEASE ANALYSIS  USING ML</span>
          <img 
            src="https://avatars.githubusercontent.com/u/124882090?v=4" 
            alt="Profile" 
            className="profile-pic"
          />
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
