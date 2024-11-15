// src/components/RightBox.jsx
import React, { useState, useEffect } from 'react';
import './RightBox.css';

const RightBox = ({ isImageUploaded }) => {
  const [analysisState, setAnalysisState] = useState('initial'); // initial, loading, complete
  const [content, setContent] = useState(null);

  useEffect(() => {
    if (isImageUploaded) {
      setAnalysisState('loading');
      // Simulate analysis delay
      setTimeout(() => {
        setAnalysisState('complete');
        setContent(getDemoContent());
      }, 4000);
    } else {
      setAnalysisState('initial');
      setContent(null);
    }
  }, [isImageUploaded]);

  const getDemoContent = () => ({
    disease: "Bacterial Leaf Blight",
    confidence: "98.5%",
    description: "Bacterial Leaf Blight is a serious disease affecting rice crops, caused by Xanthomonas oryzae pv. oryzae.",
    symptoms: [
      "Yellow to white lesions along leaf veins",
      "Lesions turn gray with age",
      "Affects crop yield significantly"
    ],
    precautions: [
      "Use disease-resistant varieties",
      "Maintain proper field drainage",
      "Apply balanced fertilizers",
      "Practice crop rotation"
    ],
    treatment: [
      "Remove infected plants",
      "Apply copper-based bactericides",
      "Maintain proper spacing between plants"
    ]
  });

  return (
    <div className="right-box">
      <div className="box-content">
        {analysisState === 'initial' && (
          <h2 className="box-title">Upload an image to see the analysis</h2>
        )}
        
        {analysisState === 'loading' && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p className="loading-text">Analyzing image...</p>
          </div>
        )}

        {analysisState === 'complete' && content && (
          <div className="analysis-content">
            <h2 className="detection-title">Detection Results</h2>
            <div className="confidence-badge">Confidence: {content.confidence}</div>
            
            <section className="disease-section">
              <h3>Disease Detected</h3>
              <p className="disease-name">{content.disease}</p>
              <p className="disease-description">{content.description}</p>
            </section>

            <section className="symptoms-section">
              <h3>Symptoms</h3>
              <ul>
                {content.symptoms.map((symptom, index) => (
                  <li key={index}>{symptom}</li>
                ))}
              </ul>
            </section>

            <section className="precautions-section">
              <h3>Precautions</h3>
              <ul>
                {content.precautions.map((precaution, index) => (
                  <li key={index}>{precaution}</li>
                ))}
              </ul>
            </section>

            <section className="treatment-section">
              <h3>Treatment</h3>
              <ul>
                {content.treatment.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ul>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default RightBox;