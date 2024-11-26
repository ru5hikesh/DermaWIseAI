//import React from "react";
import "./pages-css/About.css";


const About = () => {
  return (
    <div className="about-container">
      <h1 className="fade-in"> Skin Disease Analysis and Prediction Platform</h1>
      <p className="fade-in about-content">
        Skin diseases often go undiagnosed or misdiagnosed due to limited access to dermatological expertise.
        Our mission is to bridge this gap with technology.
      </p>

      <div className="fade-in">
        <h2>Our Journey:</h2>
        <p className="about-content">
          This platform is the culmination of rigorous research and development, combining expertise in machine
          learning, dermatology, and software engineering. By training on the globally recognized HAM10000 dataset,
          we ensure our predictions are reliable and unbiased.
        </p>
      </div>

      <div className="fade-in">
        <h2>Core Values:</h2>
        <ul className="core-values">
          <li>
            <strong>Innovation:</strong> Continually enhancing our algorithms to improve diagnostic accuracy.
          </li>
          <li>
            <strong>Accessibility:</strong> Making skin disease analysis affordable and accessible worldwide.
          </li>
          <li>
            <strong>Data Privacy:</strong> Upholding the highest standards for user confidentiality and security.
          </li>
        </ul>
      </div>

      {/* New Section: Images */}
      <div className="fade-in image-gallery">
        <h2>Our Work in Action:</h2>
        <div className="image-section">
          <img
            src="img1"
            alt="Dermatology Analysis"
            className="about-image"
          />
          <img
            src="https://via.placeholder.com/200"
            alt="Machine Learning Process"
            className="about-image"
          />
          <img
            src="https://via.placeholder.com/200"
            alt="Team Collaboration"
            className="about-image"
          />
        </div>
      </div>

      <div className="fade-in">
        <h2>The Technology:</h2>
        <p className="about-content">
          We utilize convolutional neural networks (CNNs) and ensemble models to analyze skin images. Our comparative
          study highlights how various ML/DL algorithms perform in real-world conditions.
        </p>
      </div>

      <div className="fade-in">
        <h2>Our Goal:</h2>
        <p className="about-content">
          To empower users with knowledge and tools for better skin health management.
        </p>
      </div>

      <div className="button-section">
        <button className="btn grow">Learn More</button>
        <button className="btn grow">Get Started</button>
      </div>
    </div>
  );
};

export default About;
