import  { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './pages-css/Login.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('user'); // Default to 'user'
  const [error, setError] = useState(''); // For displaying errors
  const navigate = useNavigate();

  // Function to handle form submission (Login)
  const handleLogin = async (e) => {
    e.preventDefault();

    // Validate the input fields
    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    try {
      // Send login request to API (replace with your actual API URL)
      const response = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, userType }),
      });

      const data = await response.json();

      // Handle successful login
      if (response.ok) {
        const { userType } = data; // Assuming the userType is in the response

        // Store userType in localStorage for persistence across sessions
        localStorage.setItem('userType', userType);

        // Redirect based on userType
        if (userType === 'user') {
          navigate('/User'); // Redirect to User Dashboard
        } else if (userType === 'doctor') {
          navigate('/Doctor'); // Redirect to Doctor Dashboard
        }
      } else {
        // Display error message if login fails
        setError(data.message || 'Login failed. Please try again.');
      }
    } catch (error) {
      // Handle network or other errors
      setError('An error occurred. Please try again later.');
      console.error('Error during login:', error);
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={handleLogin}>
        <h2>Login</h2>

        {/* Email Input */}
        <label>Email:</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          placeholder="Enter your email"
        />

        {/* Password Input */}
        <label>Password:</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          placeholder="Enter your password"
        />

        {/* User Type Selector */}
        <div className="user-type-selector">
          <button
            type="button"
            className={userType === 'user' ? 'active' : ''}
            onClick={() => setUserType('user')}
          >
            User
          </button>
          <button
            type="button"
            className={userType === 'doctor' ? 'active' : ''}
            onClick={() => setUserType('doctor')}
          >
            Doctor
          </button>
        </div>

        {/* Display error message */}
        {error && <div className="error-message">{error}</div>}

        {/* Login Button */}
        <button type="submit">Login</button>
      </form>
    </div>
  );
};

export default Login;
