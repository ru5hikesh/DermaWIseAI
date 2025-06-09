'use client';

import { FiShield, FiActivity, FiHeart, FiUsers, FiCode, FiAward, FiLayers, FiPhone, FiAlertTriangle, FiCheckCircle } from 'react-icons/fi';

export default function AboutPage() {
  const features = [
    {
      icon: <FiActivity className="w-6 h-6 text-blue-400" />,
      title: 'Skin Disease Classification',
      description: 'Identifies multiple skin conditions from uploaded images with high accuracy.'
    },
    {
      icon: <FiShield className="w-6 h-6 text-purple-400" />,
      title: 'AI-Powered Explanations',
      description: 'Provides detailed, easy-to-understand explanations of diagnoses.'
    },
    {
      icon: <FiLayers className="w-6 h-6 text-green-400" />,
      title: 'Advanced ML Techniques',
      description: 'Utilizes transfer learning with EfficientNetB3 and advanced training methods.'
    },
    {
      icon: <FiCheckCircle className="w-6 h-6 text-yellow-400" />,
      title: 'Robust & Reliable',
      description: 'Comprehensive error handling and fallback mechanisms.'
    }
  ];

  const conditions = [
    'Actinic keratosis',
    'Basal cell carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented benign keratosis',
    'Seborrheic keratosis',
    'Squamous cell carcinoma',
    'Vascular lesion'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0a1f] via-[#0f0f2d] to-[#1a1a3a] text-white">
      {/* Hero Section */}
      <section className="pt-24 pb-12 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <h1 className="text-3xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent 
                      bg-gradient-to-r from-blue-400 to-purple-500">
            About DermaWise AI
          </h1>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto">
            A machine learning-based skin disease detection system designed to make dermatological care accessible to everyone.
          </p>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 md:p-8">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-12 h-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full"></div>
              <h2 className="text-2xl font-bold">Our Mission</h2>
            </div>
            <p className="text-lg text-gray-300 mb-6">
              At DermaWise AI, we're on a mission to make dermatological care accessible to everyone, 
              everywhere. We combine cutting-edge artificial intelligence with medical expertise to provide 
              instant, accurate skin analysis and personalized recommendations.
            </p>
            <p className="text-lg text-gray-300">
              Our goal is to bridge the gap between patients and dermatologists, helping people understand 
              their skin conditions and when to seek professional medical advice.
            </p>
          </div>
        </div>
      </section>

      {/* Project Overview */}
      <section className="py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 md:p-8">
            <div className="flex items-center gap-4 mb-8">
              <div className="w-12 h-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full"></div>
              <h2 className="text-2xl font-bold">Project Overview</h2>
            </div>
            <p className="text-lg text-gray-300 mb-6">
              DermaWiseAI is a machine learning-based skin disease detection system that classifies various skin conditions using dermatoscopic images. 
              Designed to be accessible in resource-constrained areas, it brings expert-level dermatological analysis to your fingertips.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-12">
              <div>
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <FiLayers className="text-blue-400" />
                  Backend (Django + TensorFlow)
                </h3>
                <ul className="space-y-3 text-gray-300 pl-2">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Built with Django REST framework</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Core ML model using EfficientNetB3</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Implements singleton pattern for efficient model loading</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Integrates with AWS Bedrock for AI explanations</span>
                  </li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <FiCode className="text-purple-400" />
                  Frontend (Next.js)
                </h3>
                <ul className="space-y-3 text-gray-300 pl-2">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Modern, responsive web interface</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Built with Next.js for optimal performance</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Styled with Tailwind CSS</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Intuitive user experience</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features */}
      <section className="py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Key Features</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Advanced technology for accurate skin disease detection and analysis
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="bg-white/5 backdrop-blur-sm rounded-xl p-6 hover:bg-white/10 transition-all">
                <div className="w-12 h-12 rounded-full bg-blue-500/10 flex items-center justify-center mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Conditions */}
      <section className="py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-600/10 rounded-2xl p-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold mb-2">Supported Skin Conditions</h2>
              <p className="text-gray-400">Our model can identify various skin diseases including:</p>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {conditions.map((condition, index) => (
                <div key={index} className="flex items-center gap-3 bg-white/5 p-4 rounded-lg">
                  <FiCheckCircle className="text-green-400 flex-shrink-0" />
                  <span className="text-gray-300">{condition}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ML Methodology */}
      <section className="py-12 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8">
            <h2 className="text-2xl font-bold mb-6">Machine Learning Methodology</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold mb-4 text-blue-400">Advanced Techniques</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Transfer learning with EfficientNetB3</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Focal Loss for handling class imbalance</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-blue-400">•</span>
                    <span>Data augmentation for better generalization</span>
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-4 text-purple-400">Model Training</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Gradual unfreezing of layers</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Class weighting for imbalanced datasets</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-purple-400">•</span>
                    <span>Comprehensive model evaluation</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-12 px-4">
        <div className="max-w-4xl mx-auto text-center bg-gradient-to-r from-blue-500/10 to-purple-600/10 rounded-2xl p-8">
          <h2 className="text-2xl font-bold mb-4">Experience the Future of Dermatology</h2>
          <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
            Join thousands of users who trust DermaWise AI for their skin health needs. Get instant, accurate analysis from the comfort of your home.
          </p>
          <a href="/" className="inline-block px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 
                              text-white font-semibold rounded-lg hover:opacity-90 transition-all">
            Get Started for Free
          </a>
        </div>
      </section>
    </div>
  );
}