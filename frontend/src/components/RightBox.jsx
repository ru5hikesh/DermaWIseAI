import React from 'react';
import './RightBox.css';

function RightBox({ predictions = null }) {
  return (
    <div className="right-box">
      <div className="box-content">
        <h2 className="box-title">Predictions and Analysis of the Disease</h2>
        
        <div className="predictions-container">
          {!predictions ? (
            <div className="empty-state">
              <div className="empty-state-icon">📊</div>
              <p>Upload an image to see predictions</p>
            </div>
          ) : (
            // Example predictions - replace with actual data
            <>
              <div className="prediction-item">
                <span className="prediction-label">Disease Type 1</span>
                <div className="prediction-value">
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: '85%' }}></div>
                  </div>
                  <span>85%</span>
                </div>
              </div>

              <div className="prediction-item">
                <span className="prediction-label">Disease Type 2</span>
                <div className="prediction-value">
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: '45%' }}></div>
                  </div>
                  <span>45%</span>
                </div>
              </div>

              <div className="prediction-item">
                <span className="prediction-label">Disease Type 3</span>
                <div className="prediction-value">
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: '25%' }}></div>
                  </div>
                  <span>25%</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default RightBox;