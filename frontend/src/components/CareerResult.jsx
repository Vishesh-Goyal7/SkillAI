import React, { useState } from "react";
import "../styles/CareerResults.css";

function CareerResults({ userSkills, recommendations }) {
  const [selectedCareer, setSelectedCareer] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  const handleCardClick = (career) => {
    setSelectedCareer(career);
    setModalOpen(true);
  };

  return (
    <div className="results-container">
      <h2>Based on your selected skills, here are the best careers for you:</h2>

      <div className="user-skills">
        <h3>Your Skills:</h3>
        <ul>
          {Object.keys(userSkills)
            .filter((skill) => userSkills[skill] === 1)
            .map((skill, index) => (
              <li key={index}>{skill}</li>
            ))}
        </ul>
      </div>

      <div className="career-card-grid">
        {Object.entries(recommendations).map(([career, explanation], index) => (
          <div
            key={index}
            className="career-card"
            onClick={() => handleCardClick({ title: career, text: explanation })}
          >
            <h4>{career}</h4>
            <p>Click to read more</p>
          </div>
        ))}
      </div>

      {modalOpen && selectedCareer && (
        <div className="career-modal-overlay" onClick={() => setModalOpen(false)}>
          <div
            className="career-modal"
            onClick={(e) => e.stopPropagation()} 
          >
            <h3>{selectedCareer.title}</h3>
            <p>{selectedCareer.text}</p>
            <button onClick={() => setModalOpen(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CareerResults;
