// src/App.jsx
import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar/Navbar';
import LeftBox from './components/LeftBox/LeftBox';
import RightBox from './components/RightBox/RightBox';
import Footer from './components/Footer/Footer';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [isImageUploaded, setIsImageUploaded] = useState(false);

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
          <LeftBox onImageUpload={setIsImageUploaded} />
          <RightBox isImageUploaded={isImageUploaded} />
        </div>

        <Footer message={message} />
      </div>
    </div>
  );
}

export default App;