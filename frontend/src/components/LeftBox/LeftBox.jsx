import React, { useState } from 'react';
import './LeftBox.css';

const LeftBox = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    handleFiles(files);
  };

  const handleFiles = (files) => {
    if (files && files[0]) {
      const file = files[0];
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }

      setImage(file);
      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      handleFiles(e.target.files);
    };
    input.click();
  };

  return (
    <div
      className={`left-box ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      {previewUrl ? (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
          <button 
            className="remove-button"
            onClick={(e) => {
              e.stopPropagation();
              setImage(null);
              setPreviewUrl(null);
            }}
          >
            Remove
          </button>
        </div>
      ) : (
        <div className="box-content">
          <h2 className="box-title">Upload Image</h2>
          <p className="drag-text">
            {isDragging ? 'Drop image here' : 'Drag and drop your image here'}
          </p>
        </div>
      )}
    </div>
  );
};

export default LeftBox;