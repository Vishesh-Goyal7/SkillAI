import React from 'react';
import '../styles/Header.css';

const Header = ({ isDarkMode, toggleDarkMode }) => {

  return (
    <header className="header">
      <a className="logo" href='/'>Skill<span>AI</span></a>

      <div className="header-actions">
        <button className="toggle-button" onClick={toggleDarkMode}>
          {isDarkMode ? "🌙 Dark Mode" : "🌞 Light Mode"}
        </button>

        <a
          href="https://visheshverse.com"
          target="_blank"
          rel="noopener noreferrer"
          className="dev-button"
        >
          Meet the Developer
        </a>
      </div>
    </header>
  );
};

export default Header;
