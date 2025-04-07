'use client'

import React from 'react';

const RightBox = ({ diagnosisData, isLoading }) => {
  if (isLoading) {
    return (
      <div className="flex-1 bg-gray-900 border border-gray-300 rounded-xl h-[500px] overflow-y-auto transition-all duration-300 shadow-md p-5 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-500 border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
          <p className="mt-4 text-white text-lg">Analyzing image...</p>
        </div>
      </div>
    );
  }

  if (!diagnosisData) {
    return (
      <div className="flex-1 bg-gray-900 border border-gray-300 rounded-xl h-[500px] overflow-y-auto transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-1 p-5 flex items-center justify-center">
        <div className="text-center text-gray-400">
          <h2 className="text-2xl font-semibold mb-2">No Diagnosis Yet</h2>
          <p>Upload an image to get a skin condition diagnosis</p>
        </div>
      </div>
    );
  }

  // Format confidence as percentage
  const confidencePercentage = (diagnosisData.confidence * 100).toFixed(2);
  
  // Determine confidence level class for styling
  let confidenceClass = "text-red-500";
  if (diagnosisData.confidence > 0.7) {
    confidenceClass = "text-green-500";
  } else if (diagnosisData.confidence > 0.4) {
    confidenceClass = "text-yellow-500";
  }

  // Split explanation into paragraphs for better readability
  const explanationParagraphs = diagnosisData.explanation.split('\n\n');

  return (
    <div className="flex-1 bg-gray-900 border border-gray-300 rounded-xl h-[500px] overflow-y-auto transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-1 p-5">
      <div className="text-left">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white capitalize mb-2">{diagnosisData.disease}</h1>
          <div className="flex items-center mb-4">
            <span className="text-gray-300 mr-2">Confidence:</span>
            <span className={`font-semibold ${confidenceClass}`}>{confidencePercentage}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5 mb-6">
            <div 
              className={`h-2.5 rounded-full ${confidenceClass}`} 
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4 mb-4">
          <h2 className="text-xl font-semibold text-white mb-3">Detailed Explanation</h2>
          <div className="text-gray-300 space-y-4">
            {explanationParagraphs.map((paragraph, index) => {
              // Check if this is a section header (ends with a colon)
              if (paragraph.trim().endsWith(':')) {
                return (
                  <h3 key={index} className="text-lg font-medium text-white mt-4">
                    {paragraph}
                  </h3>
                );
              }
              
              // Handle bullet points
              if (paragraph.includes('- ')) {
                const bulletPoints = paragraph.split('\n').filter(line => line.trim());
                return (
                  <ul key={index} className="list-disc list-inside space-y-2 ml-2">
                    {bulletPoints.map((point, pointIndex) => (
                      <li key={pointIndex}>{point.replace('- ', '')}</li>
                    ))}
                  </ul>
                );
              }
              
              // Regular paragraph
              return <p key={index}>{paragraph}</p>;
            })}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-semibold text-white mb-3">When to See a Doctor</h2>
          <p className="text-gray-300">
            If you're concerned about your skin condition, please consult with a dermatologist. 
            This AI diagnosis is meant to be informative but should not replace professional medical advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default RightBox;