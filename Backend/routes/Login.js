const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const User = require('../Model/User');
const router = express.Router();
const { JWT_SECRET } = process.env;

router.post('/Login', async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ message: 'Email and password are required.' });
  }

  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: 'Invalid email or password.' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: 'Invalid email or password.' });
    }

    const token = jwt.sign({ id: user._id, userType: user.userType }, JWT_SECRET, { expiresIn: '1h' });
    res.json({ message: 'Login successful.', token, userType: user.userType });
  } catch (error) {
    console.error('Error during login:', error);
    res.status(500).json({ message: 'Server error.' });
  }
});

module.exports = router;
