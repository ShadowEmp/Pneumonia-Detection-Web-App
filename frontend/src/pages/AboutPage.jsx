import React from 'react'
import { Brain, Eye, Zap, Shield, Users, BookOpen, Github, Mail } from 'lucide-react'

function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto fade-in">
      <h1 className="text-4xl font-bold text-gray-800 mb-2 text-center">
        About This Project
      </h1>
      <p className="text-gray-600 mb-8 text-center text-lg">
        AI-Powered Pneumonia Detection System using Deep Learning and Explainable AI
      </p>

      {/* Project Overview */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Project Overview</h2>
        <p className="text-gray-700 leading-relaxed mb-4">
          This system leverages state-of-the-art deep learning techniques to automatically detect pneumonia 
          from chest X-ray images. Built with a focus on accuracy, speed, and explainability, it provides 
          healthcare professionals with a powerful tool for rapid diagnosis assistance.
        </p>
        <p className="text-gray-700 leading-relaxed">
          The system uses transfer learning with ResNet50 architecture and implements Grad-CAM 
          (Gradient-weighted Class Activation Mapping) to provide visual explanations of the model's 
          predictions, making the AI's decision-making process transparent and trustworthy.
        </p>
      </div>

      {/* Key Features */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Key Features</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center flex-shrink-0">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Deep Learning Model</h3>
              <p className="text-gray-600 text-sm">
                ResNet50-based CNN with transfer learning for high accuracy pneumonia detection
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center flex-shrink-0">
              <Eye className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Grad-CAM Visualization</h3>
              <p className="text-gray-600 text-sm">
                Visual heatmaps showing which lung regions influenced the AI's decision
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center flex-shrink-0">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Fast Processing</h3>
              <p className="text-gray-600 text-sm">
                Get prediction results in seconds with real-time image analysis
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 medical-gradient rounded-lg flex items-center justify-center flex-shrink-0">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">High Accuracy</h3>
              <p className="text-gray-600 text-sm">
                96%+ accuracy with excellent precision and recall metrics
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Technology Stack</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
              <BookOpen className="w-5 h-5 mr-2 text-blue-600" />
              Backend & AI Engine
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="font-medium text-gray-700 mb-2">Deep Learning</p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• TensorFlow / Keras</li>
                  <li>• ResNet50 (Transfer Learning)</li>
                  <li>• Grad-CAM Implementation</li>
                </ul>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="font-medium text-gray-700 mb-2">Backend API</p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Flask / FastAPI</li>
                  <li>• RESTful API Design</li>
                  <li>• Image Processing (OpenCV)</li>
                </ul>
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
              <BookOpen className="w-5 h-5 mr-2 text-teal-600" />
              Frontend & UI
            </h3>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="font-medium text-gray-700 mb-2">Framework</p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• React.js 18</li>
                  <li>• React Router</li>
                  <li>• Vite Build Tool</li>
                </ul>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="font-medium text-gray-700 mb-2">Styling & UI</p>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Tailwind CSS</li>
                  <li>• Lucide Icons</li>
                  <li>• Recharts (Visualizations)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">How It Works</h2>
        
        <div className="space-y-4">
          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
              1
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Image Upload & Preprocessing</h3>
              <p className="text-gray-600 text-sm">
                User uploads a chest X-ray image. The system preprocesses it by resizing to 224×224 pixels 
                and normalizing pixel values for optimal model input.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-teal-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
              2
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">CNN Feature Extraction</h3>
              <p className="text-gray-600 text-sm">
                The ResNet50 model extracts hierarchical features from the X-ray image through multiple 
                convolutional layers, identifying patterns associated with pneumonia.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-cyan-500 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
              3
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Binary Classification</h3>
              <p className="text-gray-600 text-sm">
                The model outputs a probability score indicating the likelihood of pneumonia. 
                A threshold of 0.5 is used to classify as Normal or Pneumonia.
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-4">
            <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center flex-shrink-0 font-bold">
              4
            </div>
            <div>
              <h3 className="font-semibold text-gray-800 mb-1">Grad-CAM Explanation</h3>
              <p className="text-gray-600 text-sm">
                Grad-CAM computes gradients of the prediction with respect to the last convolutional layer, 
                generating a heatmap that highlights important regions in the X-ray.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset Information */}
      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Dataset Information</h2>
        <p className="text-gray-700 mb-4">
          The model is trained on the <strong>Chest X-Ray Images (Pneumonia)</strong> dataset, 
          which contains thousands of labeled chest X-ray images categorized as Normal or Pneumonia.
        </p>
        <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-800 mb-2">Training Details:</h3>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>• <strong>Training Set:</strong> ~5,000 images with data augmentation</li>
            <li>• <strong>Validation Set:</strong> ~1,000 images for hyperparameter tuning</li>
            <li>• <strong>Test Set:</strong> ~1,000 images for final evaluation</li>
            <li>• <strong>Augmentation:</strong> Rotation, flipping, zoom, brightness adjustment</li>
          </ul>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="card bg-yellow-50 border-2 border-yellow-200">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
          <Shield className="w-6 h-6 mr-2 text-yellow-600" />
          Important Disclaimer
        </h2>
        <p className="text-gray-700 leading-relaxed">
          This system is designed as a <strong>diagnostic assistance tool</strong> and should not replace 
          professional medical judgment. All predictions should be reviewed by qualified healthcare 
          professionals. The system is intended for research and educational purposes.
        </p>
      </div>

      {/* Contact Section */}
      <div className="card mt-8 medical-gradient text-white">
        <h2 className="text-2xl font-bold mb-4">Get In Touch</h2>
        <p className="mb-6 text-blue-100">
          Have questions or feedback? We'd love to hear from you!
        </p>
        <div className="flex flex-wrap gap-4">
          <a href="#" className="flex items-center space-x-2 bg-white bg-opacity-20 px-4 py-2 rounded-lg hover:bg-opacity-30 transition-all">
            <Github className="w-5 h-5" />
            <span>GitHub Repository</span>
          </a>
          <a href="#" className="flex items-center space-x-2 bg-white bg-opacity-20 px-4 py-2 rounded-lg hover:bg-opacity-30 transition-all">
            <Mail className="w-5 h-5" />
            <span>Contact Us</span>
          </a>
          <a href="#" className="flex items-center space-x-2 bg-white bg-opacity-20 px-4 py-2 rounded-lg hover:bg-opacity-30 transition-all">
            <BookOpen className="w-5 h-5" />
            <span>Documentation</span>
          </a>
        </div>
      </div>
    </div>
  )
}

export default AboutPage
