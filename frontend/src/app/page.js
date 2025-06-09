'use client';

import { useState } from 'react';
import Link from 'next/link';
import { FiUpload, FiSearch, FiShield, FiClock, FiBarChart2, FiHeart } from 'react-icons/fi';

export default function Home() {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0a1f] via-[#0f0f2d] to-[#1a1a3a] text-white overflow-x-hidden">
      {/* Main Hero Section */}
      <main className="container mx-auto px-4 md:px-6 pt-20 pb-5">
        <div className="max-w-6xl mx-auto text-center">
          {/* Logo and Tagline */}
          <div className="mb-12">
            <h1 className="text-5xl md:text-7xl font-bold bg-clip-text text-transparent 
                        bg-gradient-to-r from-blue-400 to-purple-500 mb-6">
              DermaWise AI
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto">
              Advanced AI-Powered Skin Analysis at Your Fingertips
            </p>
          </div>

          {/* Main CTA */}
          <div className="max-w-2xl mx-auto mb-20">
            <div className={`relative p-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl 
                          shadow-xl transform transition-all duration-500 ${isHovered ? 'scale-105' : 'scale-100'}`}
                  onMouseEnter={() => setIsHovered(true)}
                  onMouseLeave={() => setIsHovered(false)}>
              <Link 
                href="/home"
                className="w-full bg-gray-900 hover:bg-opacity-90 text-white text-lg font-semibold 
                          py-5 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3"
              >
                <FiUpload className="text-xl" />
                Upload Skin Image for Instant Analysis
              </Link>
            </div>
            <p className="text-gray-400 mt-4 text-sm">
              Get started in seconds. No account required.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-20">
            {[
              {
                icon: <FiSearch className="w-8 h-8 mb-4 text-blue-400" />,
                title: "Accurate Diagnosis",
                desc: "Our AI analyzes skin conditions with medical-grade precision."
              },
              {
                icon: <FiClock className="w-8 h-8 mb-4 text-purple-400" />,
                title: "Instant Results",
                desc: "Get detailed analysis in seconds, not days."
              },
              {
                icon: <FiShield className="w-8 h-8 mb-4 text-green-400" />,
                title: "Secure & Private",
                desc: "Your data stays confidential and secure."
              }
            ].map((feature, index) => (
              <div key={index} className="bg-white/5 backdrop-blur-sm p-6 rounded-xl hover:bg-white/10 
                                      transition-all duration-300 hover:-translate-y-1">
                <div className="flex flex-col items-center">
                  {feature.icon}
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-gray-400">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>

          {/* How It Works */}
          <div className="max-w-4xl mx-auto mb-20">
            <h2 className="text-3xl font-bold mb-12">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                { number: "1", title: "Upload Image", desc: "Take or upload a clear photo of the affected area." },
                { number: "2", title: "AI Analysis", desc: "Our advanced algorithms analyze the image for potential conditions." },
                { number: "3", title: "Get Results", desc: "Receive instant, detailed analysis and recommendations." }
              ].map((step, index) => (
                <div key={index} className="relative group">
                  <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 
                                rounded-lg blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
                  <div className="relative bg-gray-900 p-6 rounded-lg h-full flex flex-col items-center text-center">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-xl font-bold mb-4">
                      {step.number}
                    </div>
                    <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                    <p className="text-gray-400">{step.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Final CTA */}
          <div className="max-w-2xl mx-auto">
            <h2 className="text-3xl font-bold mb-6">Ready to Take Control of Your Skin Health?</h2>
            <p className="text-xl text-gray-300 mb-8">
              Join thousands who've discovered their skin conditions early with DermaWise AI.
            </p>
            
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black/30 backdrop-blur-lg py-8 border-t border-gray-800">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">© {new Date().getFullYear()} DermaWise AI. All rights reserved.</p>
          <p className="text-sm text-gray-500 mt-2">
            This tool is for informational purposes only and is not a substitute for professional medical advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
