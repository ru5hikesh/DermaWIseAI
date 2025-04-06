// src/components/Footer.jsx
import React from 'react';

const Footer = ({ message }) => {
  return (
    <div className="bg-[#0a0a0a] text-center p-4">
      <p className="flex items-center justify-center gap-1 ">
        <span>© made with</span>
        <img
          src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Beating%20Heart.png"
          alt="Beating heart"
          className="w-4 h-4"
        />
        <span> by </span>
        <a
          href="https://github.com/ru5hikesh"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-500 font-semibold"
        >
          G26
        </a>
        <span>{message}</span>
      </p>
    </div>
  );
};

export default Footer;
