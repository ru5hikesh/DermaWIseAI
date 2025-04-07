'use client'

import React, { useState, useRef } from 'react';
import Image from 'next/image';

const LeftBox = ({ setDiagnosisData, setIsLoading }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (file.type.startsWith('image/')) {
      setSelectedImage(file);
      const fileUrl = URL.createObjectURL(file);
      setPreviewUrl(fileUrl);
      uploadImage(file);
    } else {
      alert('Please select an image file');
    }
  };

  const uploadImage = async (file) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setDiagnosisData(data);
    } catch (error) {
      console.error('Error uploading image:', error);
      alert('Error uploading image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div 
      className={`flex flex-col w-[350px] min-h-[320px] bg-gray-900 border-2 ${dragActive ? 'border-blue-500' : 'border-gray-400'} p-6 rounded-2xl shadow-lg backdrop-blur-md transition-all duration-300 ease-in-out relative overflow-hidden hover:border-gray-600 hover:-translate-y-1 hover:shadow-xl`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input 
        ref={fileInputRef}
        type="file" 
        className="hidden" 
        accept="image/*"
        onChange={handleChange}
      />

      {!previewUrl ? (
        <div className="flex flex-col items-center justify-center h-full text-center">
          <h2 className="text-white text-2xl font-bold mb-5">Upload Image</h2>
          <p className="text-gray-400 text-sm mb-4">Drag and drop your skin image here</p>
          <button 
            onClick={handleButtonClick}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300"
          >
            Browse Files
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <div className="relative w-full h-[200px] mb-4 overflow-hidden rounded-lg">
            <Image 
              src={previewUrl} 
              alt="Uploaded skin image" 
              fill
              style={{ objectFit: 'cover' }}
              className="rounded-lg"
            />
          </div>
          <button 
            onClick={handleButtonClick}
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-300 w-full"
          >
            Upload New Image
          </button>
        </div>
      )}
    </div>
  );
};

export default LeftBox;
