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

  // Get severity styling
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-500 bg-red-500/10 border-red-500';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500';
      case 'low': return 'text-green-500 bg-green-500/10 border-green-500';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500';
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'immediate': return 'text-red-500 bg-red-500/10 border-red-500';
      case 'urgent': return 'text-orange-500 bg-orange-500/10 border-orange-500';
      case 'routine': return 'text-blue-500 bg-blue-500/10 border-blue-500';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500';
    }
  };

  // Render content based on new structured format
  const renderStructuredContent = () => {
    if (!diagnosisData.sections) return null;

    return Object.entries(diagnosisData.sections).map(([key, section]) => (
      <div key={key} className="mb-6">
        <h3 className="text-xl font-semibold text-blue-400 mb-3 border-b border-gray-700 pb-2">
          {section.title}
        </h3>
        <div className="space-y-3">
          {section.content.map((contentBlock, idx) => {
            if (contentBlock.type === 'paragraph') {
              return (
                <p key={idx} className="text-gray-300">
                  {contentBlock.text}
                </p>
              );
            } else if (contentBlock.type === 'list') {
              return (
                <ol key={idx} className="list-decimal list-inside space-y-2 ml-4">
                  {contentBlock.items.map((item, itemIdx) => {
                    if (typeof item === 'string') {
                      return (
                        <li key={itemIdx} className="text-gray-300 mb-1">
                          {item}
                        </li>
                      );
                    } else {
                      return (
                        <li key={itemIdx} className="mb-2">
                          <span className="font-semibold text-gray-200">
                            {item.title}
                          </span>
                          <span className="text-gray-300 ml-1">
                            {item.description}
                          </span>
                        </li>
                      );
                    }
                  })}
                </ol>
              );
            }
            return null;
          })}
        </div>
      </div>
    ));
  };

  // Fallback to old explanation format if sections don't exist
  const renderLegacyExplanation = () => {
    if (!diagnosisData.explanation) return null;
    
    // Split the explanation by double newlines to get paragraphs
    const sections = diagnosisData.explanation.split('\n\n');
    let currentSection = null;
    const content = [];
    
    // Define known section headers
    const sectionHeaders = [
      'Symptoms:',
      'Causes:',
      'Treatment:',
      'When to See a Doctor:'
    ];
    
    // Process each paragraph
    sections.forEach((section, index) => {
      const trimmedSection = section.trim();
      if (!trimmedSection) return;
      
      // Check if this is a section header
      const isSectionHeader = sectionHeaders.some(header => 
        trimmedSection.startsWith(header)
      );
      
      if (isSectionHeader) {
        // If there was a previous section, add it to content
        if (currentSection) {
          content.push(renderSection(currentSection));
        }
        currentSection = {
          title: trimmedSection,
          content: []
        };
      } else if (currentSection) {
        // Add content to current section
        currentSection.content.push(trimmedSection);
      } else {
        // This is content before any section (introduction)
        content.push(
          <p key={`intro-${index}`} className="text-gray-300 mb-4">
            {trimmedSection}
          </p>
        );
      }
    });
    
    // Add the last section if it exists
    if (currentSection) {
      content.push(renderSection(currentSection));
    }
    
    return content;
  };
  
  // Helper function to render a section with its content (legacy format)
  const renderSection = (section) => {
    const { title, content } = section;
    const sectionId = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    
    return (
      <div key={sectionId} className="mb-6">
        <h3 className="text-xl font-semibold text-blue-400 mb-3 border-b border-gray-700 pb-2">
          {title}
        </h3>
        <div className="space-y-3">
          {content.map((paragraph, idx) => {
            // Check if paragraph is a numbered list item
            if (/^\d+\.\s/.test(paragraph)) {
              const listItems = paragraph.split('\n')
                .filter(item => item.trim())
                .map(item => item.replace(/^\d+\.\s*/, ''));
              
              return (
                <ol key={idx} className="list-decimal list-inside space-y-1 ml-4">
                  {listItems.map((item, itemIdx) => {
                    // Check for bold text (text followed by colon)
                    const parts = item.match(/^([^:]+:)(.*)$/);
                    if (parts) {
                      return (
                        <li key={itemIdx} className="mb-1">
                          <span className="font-semibold text-gray-200">{parts[1].trim()}</span>
                          {parts[2]}
                        </li>
                      );
                    }
                    return <li key={itemIdx} className="mb-1">{item}</li>;
                  })}
                </ol>
              );
            }
            
            // Regular paragraph
            // Split by bullet points if they exist
            if (paragraph.includes('- ')) {
              const bulletPoints = paragraph.split('- ').filter(point => point.trim());
              return (
                <ul key={idx} className="list-disc list-inside space-y-1 ml-4 text-gray-300">
                  {bulletPoints.map((point, pointIdx) => (
                    <li key={pointIdx}>{point.trim()}</li>
                  ))}
                </ul>
              );
            }
            
            // Regular paragraph
            return (
              <p key={idx} className="text-gray-300">
                {paragraph}
              </p>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="flex-1 bg-gray-900 border border-gray-300 rounded-xl h-[500px] overflow-y-auto transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-1 p-5">
      <div className="text-left">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white capitalize mb-2">
            {diagnosisData.disease}
          </h1>
          
          {/* Confidence Section */}
          <div className="flex items-center mb-4">
            <span className="text-gray-300 mr-2">Confidence:</span>
            <span className={`font-semibold ${confidenceClass}`}>
              {confidencePercentage}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2.5 mb-4">
            <div 
              className={`h-2.5 rounded-full ${confidenceClass}`} 
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>

          {/* Severity and Urgency Tags */}
          {(diagnosisData.severity || diagnosisData.urgency) && (
            <div className="flex gap-3 mb-4">
              {diagnosisData.severity && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getSeverityColor(diagnosisData.severity)}`}>
                  Severity: {diagnosisData.severity}
                </span>
              )}
              {diagnosisData.urgency && (
                <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getUrgencyColor(diagnosisData.urgency)}`}>
                  Urgency: {diagnosisData.urgency}
                </span>
              )}
            </div>
          )}

          {/* Summary Section */}
          {diagnosisData.summary && (
            <div className="bg-gray-800 rounded-lg p-4 mb-6">
              <h2 className="text-lg font-semibold text-white mb-2">Summary</h2>
              <p className="text-gray-300">{diagnosisData.summary}</p>
            </div>
          )}
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-semibold text-white mb-4">Detailed Analysis</h2>
          <div className="text-gray-300 space-y-4 [&_p]:text-gray-300 [&_li]:text-gray-300">
            {diagnosisData.sections ? renderStructuredContent() : renderLegacyExplanation()}
          </div>
        </div>

        {/* Disclaimer */}
        {diagnosisData.disclaimer && (
          <div className="mt-6 bg-yellow-500/10 border border-yellow-500 rounded-lg p-4">
            <p className="text-yellow-400 text-sm">
              <strong>Disclaimer:</strong> {diagnosisData.disclaimer}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default RightBox;