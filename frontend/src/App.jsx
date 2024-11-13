import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import './App.css';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/hello/')
      .then(response => response.json())
      .then(data => setMessage(data.message))
      .catch(error => console.error(error));
  }, []);

  return (
    <div className="App">
      <Navbar />
      <div className="main-content">
        <div className="boxes-container">
          <div className="left-box">
            <div className="box-content">
              <h2 className="box-title">Upload Image</h2>
              <p className="drag-text">Drag and drop your image here</p>
            </div>
          </div>
          
          <div className="right-box">
            <div className="box-content">
              <h2 className="box-title">Predictions and Analysis of the Disease</h2>
            </div>
          </div>
        </div>

        <div className="message-container">
          <h1>{message}</h1>
        </div>
      </div>
    </div>
  );
}

export default App;