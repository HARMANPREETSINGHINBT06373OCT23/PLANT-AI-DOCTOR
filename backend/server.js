require('dotenv').config({ path: __dirname + '/.env' }); // ✅ Force-load .env file
console.log("✅ MONGO_URI from env:", process.env.MONGO_URI); // Debug

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('✅ MongoDB connected'))
  .catch(err => console.error('❌ MongoDB connection error:', err.message));

app.use('/api', require('./routes/auth'));
app.use('/api/admin', require('./routes/admin'));

app.listen(3000, () => console.log('🚀 Server running on port 3000'));
