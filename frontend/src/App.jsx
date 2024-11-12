import React, { useEffect, useState } from 'react';

function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    // Replace with your Django API URL
    fetch('http://127.0.0.1:8000/api/hello/')
      .then(response => response.json())
      .then(data => setMessage(data.message))
      .catch(error => console.error(error));
  }, []);

  return (
    <div className="App">
      <h1>{message}</h1>
    </div>
  );
}

export default App;
