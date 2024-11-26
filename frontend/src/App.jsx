import  { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import Footer from './components/Dashboard/Footer/Footer';
import Home from './components/Pages/Home';
import About from './components/Pages/About';
import Contact from './components/Pages/Contact';
import Service from './components/Pages/Service';
import Login from './components/Pages/Login';
import Register from './components/Pages/Register';
import User from './components/Dashboard/User/User';
import Doctor from './components/Dashboard/Doctor/Doctor';
import LeftBox from './components/Dashboard/LeftBox/LeftBox';
import RightBox from './components/Dashboard/RightBox/RightBox';
import './App.css';

function App() {
  const [message, setMessage] = useState('');
  const [isImageUploaded, setIsImageUploaded] = useState(false);
  const [userType, setUserType] = useState(null); // 'user' or 'doctor'

  // Fetch message from the backend
  useEffect(() => {
    fetch('http://localhost:5000/')
      .then(response => response.json())
      .then(data => setMessage(data.message))
      .catch(error => console.error('Error fetching API:', error));

    // Check if the user is logged in and set the user role
    const storedUserType = localStorage.getItem('userType');
    if (storedUserType) {
      setUserType(storedUserType); // Set the user type (user/doctor)
    }
  }, []);

  return (
    <Router>
      <div className="App">
        <Navbar />

        <Routes>
          {/* Landing page or main content */}
          <Route 
            path="/" 
            element={
              <div className="main-content">
                <div className="boxes-container">
                  <LeftBox onImageUpload={setIsImageUploaded} />
                  <RightBox isImageUploaded={isImageUploaded} />
                </div>
                <Footer message={message} />
              </div>
            } 
          />
          
          {/* Public routes */}
          <Route path="/Home" element={<Home />} />
          <Route path="/About" element={<About />} />
          <Route path="/Contact" element={<Contact />} />
          <Route path="/Service" element={<Service />} />
          <Route path="/Login" element={<Login />} />
          <Route path="/Register" element={<Register />} />
          
          {/* Protected routes for authenticated users */}
          {userType && (
            <>
              <Route path="/User" element={<User />} />
              <Route path="/Doctor" element={<Doctor />} />
            </>
          )}
        </Routes>
      </div>
    </Router>
  );
}

export default App;
