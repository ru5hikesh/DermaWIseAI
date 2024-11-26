const express = require('express');
const bcrypt = require('bcrypt');
const User = require('../Model/User'); // Adjust path to your User model
const router = express.Router();

// Register route
router.post('/Register', async (req, res) => {
  const { name, email, password, userType, specialization } = req.body;

  // Basic validation
  if (!name || !email || !password) {
    return res.status(400).json({ message: 'All fields are required.' });
  }

  try {
    // Check if the user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists.' });
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create and save the new user
    const newUser = new User({
      name,
      email,
      password: hashedPassword,
      userType,
      specialization: userType === 'doctor' ? specialization : undefined,
    });

    await newUser.save();
    res.status(201).json({ message: 'User registered successfully.' });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ message: 'Server error. Please try again.' });
  }
});

module.exports = router;
