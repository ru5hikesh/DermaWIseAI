'use client';

import { useState } from 'react';
import { FiMail, FiPhone, FiMapPin, FiSend, FiCheckCircle } from 'react-icons/fi';

const ContactPage = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user types
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    if (!formData.name.trim()) newErrors.name = 'Name is required';
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    if (!formData.subject.trim()) newErrors.subject = 'Subject is required';
    if (!formData.message.trim()) newErrors.message = 'Message is required';
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setIsSubmitting(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Reset form
      setFormData({
        name: '',
        email: '',
        subject: '',
        message: ''
      });
      
      setIsSubmitted(true);
      setTimeout(() => setIsSubmitted(false), 5000);
    } catch (error) {
      console.error('Error submitting form:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const contactInfo = [
    {
      icon: <FiMail className="w-6 h-6 text-blue-400" />,
      title: 'Email Us',
      content: 'contact@dermawiseai.com',
      link: 'mailto:contact@dermawiseai.com'
    },
    {
      icon: <FiPhone className="w-6 h-6 text-blue-400" />,
      title: 'Call Us',
      content: '+91 9420473469',
      link: 'tel:+91 9420473469'
    },
    {
      icon: <FiMapPin className="w-6 h-6 text-blue-400" />,
      title: 'Visit Us',
      content: '27/A/1/2C, Varale, Talegaon Dabhade, Maharashtra 410507',
      link: 'https://maps.app.goo.gl/9TFB6QzKcMKBM5Ta7'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            Get In Touch
          </h1>
          <p className="text-lg text-gray-300 max-w-2xl mx-auto">
            Have questions or feedback? We'd love to hear from you. Reach out to us using the form below or through our contact information.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-start">
          {/* Contact Form */}
          <div className="bg-gray-800 bg-opacity-50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl border border-gray-700">
            <h2 className="text-2xl font-bold mb-6">Send us a message</h2>
            
            {isSubmitted ? (
              <div className="bg-green-900 bg-opacity-30 border border-green-700 text-green-300 p-6 rounded-xl text-center">
                <FiCheckCircle className="w-12 h-12 mx-auto mb-4 text-green-400" />
                <h3 className="text-xl font-semibold mb-2">Message Sent Successfully!</h3>
                <p>Thank you for contacting us. We'll get back to you soon.</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-300 mb-2">
                    Full Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                      errors.name ? 'border-red-500' : 'border-gray-600'
                    } rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200`}
                    placeholder="John Doe"
                  />
                  {errors.name && <p className="mt-1 text-sm text-red-400">{errors.name}</p>}
                </div>

                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
                    Email Address <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                      errors.email ? 'border-red-500' : 'border-gray-600'
                    } rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200`}
                    placeholder="you@example.com"
                  />
                  {errors.email && <p className="mt-1 text-sm text-red-400">{errors.email}</p>}
                </div>

                <div>
                  <label htmlFor="subject" className="block text-sm font-medium text-gray-300 mb-2">
                    Subject <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                      errors.subject ? 'border-red-500' : 'border-gray-600'
                    } rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200`}
                    placeholder="How can we help?"
                  />
                  {errors.subject && <p className="mt-1 text-sm text-red-400">{errors.subject}</p>}
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-300 mb-2">
                    Your Message <span className="text-red-500">*</span>
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    rows="5"
                    value={formData.message}
                    onChange={handleChange}
                    className={`w-full px-4 py-3 bg-gray-700 bg-opacity-50 border ${
                      errors.message ? 'border-red-500' : 'border-gray-600'
                    } rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200`}
                    placeholder="Type your message here..."
                  ></textarea>
                  {errors.message && <p className="mt-1 text-sm text-red-400">{errors.message}</p>}
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-300 flex items-center justify-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Sending...
                    </>
                  ) : (
                    <>
                      <FiSend className="text-lg" />
                      Send Message
                    </>
                  )}
                </button>
              </form>
            )}
          </div>

          {/* Contact Information */}
          <div className="space-y-8">
            <div className="space-y-6">
              <h2 className="text-2xl font-bold">Contact Information</h2>
              <p className="text-gray-300">
                Have questions about DermaWise AI or need support? We're here to help. Reach out to us through any of these channels.
              </p>
              
              <div className="space-y-6">
                {contactInfo.map((item, index) => (
                  <a 
                    key={index} 
                    href={item.link} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="flex items-start space-x-4 p-4 bg-gray-800 bg-opacity-50 rounded-xl hover:bg-opacity-70 transition duration-200 border border-gray-700 hover:border-blue-500"
                  >
                    <div className="p-2 bg-gray-700 rounded-lg">
                      {item.icon}
                    </div>
                    <div>
                      <h3 className="font-semibold text-blue-400">{item.title}</h3>
                      <p className="text-gray-300">{item.content}</p>
                    </div>
                  </a>
                ))}
              </div>
            </div>

            <div className="pt-6 border-t border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Our Location</h3>
              <div className="aspect-w-16 aspect-h-9 rounded-xl overflow-hidden border border-gray-700">
                <iframe
                  src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3774.209385354042!2d73.6856484!3d18.7471797!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bc2b30672e789d3%3A0x576e22fb0b7d4c64!2sDr.%20D%2BY%20Patil%20College%20of%20Engineering%20%26%20Innovation!5e0!3m2!1sen!2sin!4v1620000000000!5m2!1sen!2sin"
                  width="100%"
                  height="300"
                  style={{ border: 0 }}
                  allowFullScreen=""
                  loading="lazy"
                  className="w-full h-64 rounded-lg"
                ></iframe>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* FAQ Section */}
      <div className="bg-gray-900 py-16">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
              Frequently Asked Questions
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Find answers to common questions about DermaWise AI and our services.
            </p>
          </div>
          
          <div className="max-w-3xl mx-auto space-y-4">
            {[
              {
                question: "How accurate is DermaWise AI's skin disease detection?",
                answer: "DermaWise AI uses advanced machine learning models with an accuracy of over 90% for most common skin conditions. However, it's important to note that this is not a substitute for professional medical diagnosis."
              },
              {
                question: "Is my personal health data secure?",
                answer: "Yes, we take data security and privacy very seriously. All health data is encrypted in transit and at rest, and we comply with all relevant data protection regulations."
              },
              {
                question: "How long does it take to get a response to my inquiry?",
                answer: "Our team typically responds to inquiries within 24-48 hours during business days. For urgent matters, please call our support line."
              },
              {
                question: "Can I use DermaWise AI for emergency medical situations?",
                answer: "No, DermaWise AI is not intended for emergency medical situations. If you're experiencing a medical emergency, please contact your local emergency services immediately."
              }
            ].map((faq, index) => (
              <div key={index} className="bg-gray-800 bg-opacity-50 rounded-xl p-6">
                <h3 className="font-semibold text-lg text-blue-400 mb-2">{faq.question}</h3>
                <p className="text-gray-300">{faq.answer}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContactPage;