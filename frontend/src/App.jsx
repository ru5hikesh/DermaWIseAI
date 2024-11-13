import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import LeftBox from './components/LeftBox'; // Import LeftBox
import RightBox from './components/RightBox'; // Import RightBox
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
        <div className="boxes">
          <LeftBox />
          <RightBox />
        </div>
        <div className="message-container">
          <h1>{message}</h1>
        </div>
      </div>
    </div>
  );
}

export default App;
