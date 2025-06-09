// src/components/Navbar.jsx
"use client"
import React from 'react';

function Navbar() {
  return (
    <nav className="bg-gray-900 py-2 px-5 fixed top-0 left-0 w-full shadow-md z-50">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <ul className="flex gap-5 m-0 p-0 list-none">
          <li><a href="about" className="text-white text-base font-medium px-4 py-2 rounded-md transition duration-300 hover:bg-gray-600 hover:text-gray-200">About</a></li>
          <li><a href="services" className="text-white text-base font-medium px-4 py-2 rounded-md transition duration-300 hover:bg-gray-600 hover:text-gray-200">Services</a></li>
          <li><a href="contact" className="text-white text-base font-medium px-4 py-2 rounded-md transition duration-300 hover:bg-gray-600 hover:text-gray-200">Contact</a></li>
        </ul>

        <div className="flex items-center gap-3">
          <span className="text-white text-base font-medium">ru5hikesh</span>
          <img 
            src="https://avatars.githubusercontent.com/u/124882090?v=4" 
            alt="Profile"
            className="w-10 h-10 rounded-full object-cover border-2 border-white"
          />
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
