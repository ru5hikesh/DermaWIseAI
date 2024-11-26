// src/components/Footer.jsx
import React from 'react';
import './Footer.css'; 

const Footer = ({ message }) => {
  return (
    <div className="footer">
      <p>
        <span>© made with </span>
        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Beating%20Heart.png" alt="Beating heart" />
        <span> by </span>
        <a href="https://github.com/ru5hikesh" target="_blank" rel="noopener noreferrer">G26</a>
        <span> {message} </span>
      </p>
    </div>
  );
}

export default Footer;
