// src/components/LeftBox.jsx
import React, { useState } from 'react';
import './LeftBox.css';

const LeftBox = ({ onImageUpload }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleDragEvents = (e, isDragging) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(isDragging);
  };

  const handleFiles = (files) => {
    if (files?.[0]) {
      const file = files[0];
      if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result);
        onImageUpload(true); // Trigger the analysis in RightBox
      };
      reader.readAsDataURL(file);
    }
  };

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => handleFiles(e.target.files);
    input.click();
  };

  return (
    <div
      className={`left-box ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
      onDragEnter={(e) => handleDragEvents(e, true)}
      onDragLeave={(e) => handleDragEvents(e, false)}
      onDragOver={(e) => handleDragEvents(e, true)}
      onDrop={(e) => {
        handleDragEvents(e, false);
        handleFiles(e.dataTransfer.files);
      }}
      onClick={handleClick}
    >
      {previewUrl ? (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
          <button 
            className="remove-button"
            onClick={(e) => {
              e.stopPropagation();
              setPreviewUrl(null);
              onImageUpload(false);
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