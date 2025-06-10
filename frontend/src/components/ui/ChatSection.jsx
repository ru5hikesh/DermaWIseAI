'use client';

import { useState } from 'react';

const ChatSection = ({ disease }) => {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiResponse, setApiResponse] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim() || !disease) return;

    setIsLoading(true);
    setApiResponse('');

    try {
      // Ensure the disease name is included in every message
      const fullMessage = message.startsWith('[') 
        ? message 
        : `[About ${disease}] ${message}`;

      const response = await fetch('http://127.0.0.1:8000/api/bedrock/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          disease: `${disease}.`,
          Rules: `Make sure it's a small answer. You are a dermatology expert. Respond in simple language, only related to the specified disease. Do not ask for images or visual input. Respond in a maximum of 2 short lines. Avoid general advice — just assess if the disease is harmful or not and briefly explain why. `,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from chat API');
      }

      const data = await response.json();
      // Extract the text content from the response
      const responseText = data.content?.[0]?.text || 'No response from the assistant.';
      setApiResponse(responseText);
    } catch (error) {
      console.error('Error:', error);
      setApiResponse('Sorry, there was an error processing your request.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-300 rounded-xl overflow-hidden transition-all duration-300 shadow-md mt-6">
      <div className="p-5">
        <h2 className="text-2xl font-semibold text-white mb-4">Ask about {disease}</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder={`Ask me anything about ${disease}...`}
              className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading || !message.trim()}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Sending...' : 'Send Message'}
          </button>
        </form>

        {apiResponse && (
          <div className="mt-6 pt-4 border-t border-gray-700">
            <h3 className="text-lg font-medium text-white mb-3">Response</h3>
            <div 
              className="prose prose-invert max-w-none text-gray-300"
              dangerouslySetInnerHTML={{ __html: apiResponse }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatSection;
