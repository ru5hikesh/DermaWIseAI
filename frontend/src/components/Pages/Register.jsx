import React, { useState } from 'react';
import './pages-css/Register.css';

function Register() {
  const [userType, setUserType] = useState('user'); // 'user' or 'doctor'
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    specialization: '', // Field for doctor's specialization
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (formData.password !== formData.confirmPassword) {
      alert("Passwords don't match!");
      return;
    }
  
    try {
      const response = await fetch('http://localhost:5000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          email: formData.email,
          password: formData.password,
          userType,
          specialization: userType === 'doctor' ? formData.specialization : '',
        }),
      });
  
      const result = await response.json();
      if (response.ok) {
        alert(result.message);
      } else {
        alert(result.error);
      }
    } catch (err) {
      console.error('Error during registration:', err);
      alert('Server error');
    }
  };
  
  return (
    <div className="register-page">
      <div className="register-container">
        <h2>Register</h2>
        <div className="user-type-selector">
          <button
            className={userType === 'user' ? 'active' : ''}
            onClick={() => setUserType('user')}
          >
            User
          </button>
          <button
            className={userType === 'doctor' ? 'active' : ''}
            onClick={() => setUserType('doctor')}
          >
            Doctor
          </button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Name</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              required
            />
          </div>
          <div className="form-group">
            <label htmlFor="confirmPassword">Confirm Password</label>
            <input
              type="password"
              id="confirmPassword"
              name="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleInputChange}
              required
            />
          </div>

          {/* Conditionally render specialization field if userType is 'doctor' */}
          {userType === 'doctor' && (
            <div className="form-group">
              <label htmlFor="specialization">Doctor`s Specialization</label>
              <input
                type="text"
                id="specialization"
                name="specialization"
                value={formData.specialization}
                onChange={handleInputChange}
                placeholder="e.g., Dermatologist"
                required
              />
            </div>
          )}

          <button type="submit" className="register-button">
            Register as {userType.charAt(0).toUpperCase() + userType.slice(1)}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Register;
