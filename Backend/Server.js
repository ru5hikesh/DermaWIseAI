
/*
const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');
const cors = require('cors');

// Initialize the app
const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());

// MongoDB Connection
const MONGO_URI ='mongodb+srv://computerdeapartmentms903:9hGtxyH6xZNCqCH9@cluster1.z9por.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1';

mongoose
  .connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  userType: { type: String, enum: ['user', 'doctor'], required: true },
  specialization: { type: String, default: '' },
});

// User Model
const User = mongoose.model('User', userSchema);

// Secret key for JWT
const JWT_SECRET = 'your_secret_key_here';

// Middleware for Authentication
const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1]; // Extract token from Authorization header
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded; // Attach user information to request object
    next();
  } catch (err) {
    return res.status(401).json({ message: 'Invalid token.' });
  }
};

// Routes

// Test Route
app.get('/', (req, res) => {
  res.send('Server is running');
});

// Register Route
app.post('/register', async (req, res) => {
  const { name, email, password, userType, specialization } = req.body;

  // Input Validation
  if (!name || !email || !password || !userType) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  if (userType === 'doctor' && !specialization) {
    return res.status(400).json({ error: 'Specialization is required for doctors' });
  }

  try {
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email is already registered' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create and save new user
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
      userType,
      specialization: userType === 'doctor' ? specialization : '',
    });

    await newUser.save();
    res.status(201).json({ message: 'User registered successfully' });
  } catch (err) {
    console.error('Error during registration:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Login Route
app.post('/login', async (req, res) => {
  const { email, password, userType } = req.body;

  // Validate input
  if (!email || !password || !userType) {
    return res.status(400).json({ message: 'All fields are required' });
  }

  try {
    // Check if user exists
    const user = await User.findOne({ email, userType });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Invalid password' });
    }

    // Generate JWT token
    const token = jwt.sign({ userId: user._id, userType: user.userType }, JWT_SECRET, {
      expiresIn: '1h',
    });

    res.status(200).json({
      message: 'Login successful',
      userType: user.userType,
      token,
    });
  } catch (err) {
    console.error('Error during login:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// User Dashboard Route
app.get('/api/user', authenticate, async (req, res) => {
  try {
    // Fetch user details from the database
    const user = await User.findById(req.user.userId).select('-password'); // Exclude password
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.status(200).json(user);
  } catch (err) {
    console.error('Error fetching user details:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start the Server
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
*/

const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const bodyParser = require('body-parser');
const cors = require('cors');

// Initialize the app
const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());

// MongoDB Connection
const MONGO_URI =
  'mongodb+srv://computerdeapartmentms903:9hGtxyH6xZNCqCH9@cluster1.z9por.mongodb.net/?retryWrites=true&w=majority&appName=Cluster1';

mongoose
  .connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('MongoDB connection error:', err));

// User Schema
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  userType: { type: String, enum: ['user', 'doctor'], required: true },
  specialization: { type: String, default: '' },
});

// User Model
const User = mongoose.model('User', userSchema);

// Secret key for JWT
const JWT_SECRET = 'your_secret_key_here';

// Middleware for Authentication
const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1]; // Extract token from Authorization header
  if (!token) {
    return res.status(401).json({ message: 'Access denied. No token provided.' });
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded; // Attach user information to request object
    next();
  } catch (err) {
    return res.status(401).json({ message: 'Invalid token.' });
  }
};

// Routes

// Test Route
app.get('/', (req, res) => {
  res.send('Server is running');
});

// Register Route
app.post('/register', async (req, res) => {
  const { name, email, password, userType, specialization } = req.body;

  // Input Validation
  if (!name || !email || !password || !userType) {
    return res.status(400).json({ error: 'All fields are required' });
  }

  if (userType === 'doctor' && !specialization) {
    return res.status(400).json({ error: 'Specialization is required for doctors' });
  }

  try {
    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: 'Email is already registered' });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create and save new user
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
      userType,
      specialization: userType === 'doctor' ? specialization : '',
    });

    await newUser.save();
    res.status(201).json({ message: 'User registered successfully' });
  } catch (err) {
    console.error('Error during registration:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Login Route
app.post('/login', async (req, res) => {
  const { email, password, userType } = req.body;

  // Validate input
  if (!email || !password || !userType) {
    return res.status(400).json({ message: 'All fields are required' });
  }

  try {
    // Check if user exists
    const user = await User.findOne({ email, userType });
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Invalid password' });
    }

    // Generate JWT token
    const token = jwt.sign({ userId: user._id, userType: user.userType }, JWT_SECRET, {
      expiresIn: '1h',
    });

    res.status(200).json({
      message: 'Login successful',
      userType: user.userType,
      token,
    });
  } catch (err) {
    console.error('Error during login:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// User Dashboard Route
app.get('/api/user', authenticate, async (req, res) => {
  try {
    // Fetch user details from the database
    const user = await User.findById(req.user.userId).select('-password'); // Exclude password
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.status(200).json(user);
  } catch (err) {
    console.error('Error fetching user details:', err);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start the Server
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
