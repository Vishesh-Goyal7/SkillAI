import React, { useEffect, useState } from 'react';
import axios from 'axios';
import baseURL from '../utils/baseURL';
import CareerResults from './CareerResult';
import '../styles/SkillSelector.css';

const SkillSelector = () => {
  const [skills, setSkills] = useState([]);
  const [selected, setSelected] = useState([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get(`${baseURL}/list`)
      .then(res => {
        setSkills(res.data.columns || res.data || []);
      })
      .catch(err => {
        alert("Failed to fetch skill list.");
        console.error(err);
      });
  }, []);

  const toggleSkill = (skill) => {
    setSelected(prev =>
      prev.includes(skill) ? prev.filter(s => s !== skill) : [...prev, skill]
    );
  };

  const handleSubmit = async () => {
    setLoading(true);
    const payload = {};
    skills.forEach(skill => {
      payload[skill] = selected.includes(skill) ? 1 : 0;
    });

    try {
      await axios.post(`${baseURL}/input`, { user_skills: payload });
      const res = await axios.get(`${baseURL}/process`);
      setResults(res.data);
    } catch (err) {
      alert("Prediction failed.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filteredSkills = skills.filter(skill =>
    skill.toLowerCase().includes(query.toLowerCase())
  );

  // 👉 If results available, switch to CareerResults view
  if (results) {
    const selectedMap = {};
    skills.forEach(skill => {
      selectedMap[skill] = selected.includes(skill) ? 1 : 0;
    });

    return (
      <CareerResults
        userSkills={selectedMap}
        recommendations={results}
      />
    );
  }

  return (
    <div className="selector-container">
      <h2>Choose Your Skills</h2>
      <input
        type="text"
        placeholder="Search skills..."
        value={query}
        onChange={e => setQuery(e.target.value)}
      />

      <div className="bubble-container">
        {filteredSkills.map(skill => (
          <div
            key={skill}
            className={`bubble ${selected.includes(skill) ? 'selected' : ''}`}
            onClick={() => toggleSkill(skill)}
          >
            {skill}
          </div>
        ))}
      </div>

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Processing...' : 'Start Searching'}
      </button>
    </div>
  );
};

export default SkillSelector;
