import React from 'react';
import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <ul>
          <li><a href="home">Home</a></li>
          <li><a href="about">About</a></li>
          <li><a href="services">Services</a></li>
          <li><a href="contact">Contact</a></li>
        </ul>
        
        <div className="profile-section">
          <span className="profile-name">ru5hikesh</span>
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