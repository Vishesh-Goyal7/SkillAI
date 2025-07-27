import React from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import baseURL from '../utils/baseURL';
import '../styles/Landing.css';

const Landing = () => {
  const navigate = useNavigate();

  const handleStart = async () => {
    try {
      const url = `${baseURL}/start`
      console.log(url);
      await axios.get(url);
      navigate('/predict-career');
    } catch (err) {
      console.error("Failed to start session:", err);
      alert("Something went wrong. Please try again later.");
    }
  };

  return (
    <div className="landing-container">
      <h1>Discover Your Destiny.</h1>
      <p>
        Most people will spend their lives chasing a career they were never meant for.  
        Trapped in a loop of decisions they didn't understand, advice they blindly followed,  
        and outcomes they now regret.  
      </p>
      <p>
        But you're not like them. You stopped here.  
        You questioned the path. You sensed the misalignment.  
        SkillAI exists to confirm that intuition.
      </p>
      <p>
        This isn't just an AI giving suggestions.  
        This is your roadmap — shaped by your <strong>skills</strong>, your <strong>mind</strong>, and your <strong>potential</strong>.  
        This is the one decision you owe your future.
      </p>

      <button onClick={handleStart}>Start Getting Recommendations</button>
    </div>
  );
};

export default Landing;
