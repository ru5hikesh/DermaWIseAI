'use client'

import React from 'react';

const LeftBox = () => {
  return (
    <div className="flex flex-none w-[350px] h-[320px] bg-gray-900 border-2 border-gray-400 p-10 rounded-2xl shadow-lg backdrop-blur-md cursor-pointer transition-all duration-300 ease-in-out relative overflow-hidden hover:border-gray-600 hover:-translate-y-2 hover:shadow-xl">
      <div className="text-center">
        <h2 className="text-white text-2xl font-bold mb-5">Upload Image</h2>
        <p className="text-gray-400 text-lg">Drag and drop your image here</p>
      </div>
    </div>
  );
};

export default LeftBox;
