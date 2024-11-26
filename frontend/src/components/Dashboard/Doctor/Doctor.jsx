import React, { useState } from 'react';
import axios from 'axios';
import './Doctor.css';

const Doctor = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    specialization: '',
    userType: 'doctor', // Fixed as 'doctor'
  });

  const [loginData, setLoginData] = useState({
    email: '',
    password: '',
  });

  const [doctorDetails, setDoctorDetails] = useState(null);
  const [token, setToken] = useState('');

  const handleRegisterChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleLoginChange = (e) => {
    setLoginData({ ...loginData, [e.target.name]: e.target.value });
  };

  const registerDoctor = async () => {
    try {
      const response = await axios.post('http://localhost:5000/register', formData);
      alert(response.data.message);
    } catch (error) {
      alert(error.response?.data?.error || 'Registration failed');
    }
  };

  const loginDoctor = async () => {
    try {
      const response = await axios.post('http://localhost:5000/login', {
        ...loginData,
        userType: 'doctor', // Ensure the type is doctor for login
      });
      setToken(response.data.token);
      alert('Login successful!');
    } catch (error) {
      alert(error.response?.data?.message || 'Login failed');
    }
  };

  const fetchDoctorDetails = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/doctor', {
        headers: { Authorization: `Bearer ${token}` },
      });
      setDoctorDetails(response.data);
    } catch (error) {
      alert(error.response?.data?.message || 'Failed to fetch details');
    }
  };

  return (
    <div>
      <h1>Doctor Portal</h1>

      <div>
        <h2>Register as Doctor</h2>
        <input
          type="text"
          name="name"
          placeholder="Name"
          value={formData.name}
          onChange={handleRegisterChange}
        />
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleRegisterChange}
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleRegisterChange}
        />
        <input
          type="text"
          name="specialization"
          placeholder="Specialization"
          value={formData.specialization}
          onChange={handleRegisterChange}
        />
        <button onClick={registerDoctor}>Register</button>
      </div>

      <div>
        <h2>Login as Doctor</h2>
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={loginData.email}
          onChange={handleLoginChange}
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={loginData.password}
          onChange={handleLoginChange}
        />
        <button onClick={loginDoctor}>Login</button>
      </div>

      <div>
        <h2>Doctor Details</h2>
        <button onClick={fetchDoctorDetails}>Fetch My Details</button>
        {doctorDetails && (
          <div>
            <p><strong>Name:</strong> {doctorDetails.name}</p>
            <p><strong>Email:</strong> {doctorDetails.email}</p>
            <p><strong>Specialization:</strong> {doctorDetails.specialization}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Doctor;
