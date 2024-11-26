
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './User.css';

const User = () => {
  const [userDetails, setUserDetails] = useState(null); // To store user details
  const navigate = useNavigate();

  // Fetch user data on component load
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          navigate('/login'); // Redirect to login if no token
          return;
        }

        const response = await fetch('http://localhost:5000/api/user', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
        });

        if (response.ok) {
          const data = await response.json();
          setUserDetails(data);
        } else {
          // Handle unauthorized or other errors
          localStorage.removeItem('token');
          navigate('/login');
        }
      } catch (err) {
        console.error('Error fetching user data:', err);
        navigate('/login');
      }
    };

    fetchUserData();
  }, [navigate]);

  // Logout function
  const handleLogout = () => {
    localStorage.removeItem('token');
    navigate('/login');
  };

  if (!userDetails) {
    return <div>Loading...</div>; // Show loading state while fetching data
  }

  return (
    <div className="user-dashboard">
      <h1>Welcome, {userDetails.name}!</h1>
      <p>Email: {userDetails.email}</p>
      <p>User Type: {userDetails.userType}</p>
      <button className="logout-button" onClick={handleLogout}>
        Logout
      </button>
    </div>
  );
};

export default User;
