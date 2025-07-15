const express = require('express');
const User = require('../models/user');
const router = express.Router();

// List all users (without passwords)
router.get('/users', async (req, res) => {
  try {
    const users = await User.find({}, '-password');
    res.json({ count: users.length, users });
  } catch (err) {
    res.status(500).json({ error: 'Failed to fetch users' });
  }
});

module.exports = router;
