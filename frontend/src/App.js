import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Landing from './components/Landing';
import SkillSelector from './components/SkillSelector';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  return (
    <Router>
      <Header isDarkMode={isDarkMode} toggleDarkMode={() => setIsDarkMode(!isDarkMode)} />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/predict-career" element={<SkillSelector />} />
      </Routes>
    </Router>
  );
}

export default App;
