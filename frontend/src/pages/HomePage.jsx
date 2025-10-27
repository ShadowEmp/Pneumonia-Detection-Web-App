import React from 'react'
import { Link } from 'react-router-dom'
import { Upload, Brain, Eye, TrendingUp, Shield, Zap } from 'lucide-react'

function HomePage() {
  return (
    <div className="fade-in">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-5xl font-bold text-gray-800 mb-4">
          AI-Powered Pneumonia Detection
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Advanced deep learning system for accurate pneumonia detection from chest X-ray images
          with explainable AI visualization using Grad-CAM
        </p>
        <Link to="/upload">
          <button className="btn-primary text-lg px-8 py-4 inline-flex items-center space-x-3">
            <Upload className="w-6 h-6" />
            <span>Upload X-Ray Image</span>
          </button>
        </Link>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-3 gap-8 mb-16">
        <div className="card text-center">
          <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center mx-auto mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">Deep Learning</h3>
          <p className="text-gray-600">
            Powered by ResNet50 transfer learning with state-of-the-art accuracy
          </p>
        </div>

        <div className="card text-center">
          <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center mx-auto mb-4">
            <Eye className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">Explainable AI</h3>
          <p className="text-gray-600">
            Grad-CAM visualization shows exactly which lung regions influenced the prediction
          </p>
        </div>

        <div className="card text-center">
          <div className="w-16 h-16 medical-gradient rounded-full flex items-center justify-center mx-auto mb-4">
            <Zap className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-bold text-gray-800 mb-2">Fast & Accurate</h3>
          <p className="text-gray-600">
            Get results in seconds with high precision and recall metrics
          </p>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="card mb-16">
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">How It Works</h2>
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-12 h-12 bg-blue-500 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-xl font-bold">
              1
            </div>
            <h4 className="font-semibold text-gray-800 mb-2">Upload Image</h4>
            <p className="text-sm text-gray-600">
              Upload a chest X-ray image (PNG, JPG, JPEG)
            </p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-teal-500 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-xl font-bold">
              2
            </div>
            <h4 className="font-semibold text-gray-800 mb-2">AI Analysis</h4>
            <p className="text-sm text-gray-600">
              Deep learning model analyzes the image
            </p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-cyan-500 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-xl font-bold">
              3
            </div>
            <h4 className="font-semibold text-gray-800 mb-2">Get Results</h4>
            <p className="text-sm text-gray-600">
              Receive prediction with confidence score
            </p>
          </div>

          <div className="text-center">
            <div className="w-12 h-12 bg-blue-600 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-xl font-bold">
              4
            </div>
            <h4 className="font-semibold text-gray-800 mb-2">View Explanation</h4>
            <p className="text-sm text-gray-600">
              See Grad-CAM heatmap highlighting key regions
            </p>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="grid md:grid-cols-3 gap-8 mb-16">
        <div className="card text-center bg-gradient-to-br from-blue-50 to-blue-100">
          <TrendingUp className="w-12 h-12 text-blue-600 mx-auto mb-3" />
          <h3 className="text-4xl font-bold text-blue-600 mb-2">95%+</h3>
          <p className="text-gray-700 font-medium">Accuracy</p>
        </div>

        <div className="card text-center bg-gradient-to-br from-teal-50 to-teal-100">
          <Shield className="w-12 h-12 text-teal-600 mx-auto mb-3" />
          <h3 className="text-4xl font-bold text-teal-600 mb-2">Fast</h3>
          <p className="text-gray-700 font-medium">Instant Results</p>
        </div>

        <div className="card text-center bg-gradient-to-br from-cyan-50 to-cyan-100">
          <Eye className="w-12 h-12 text-cyan-600 mx-auto mb-3" />
          <h3 className="text-4xl font-bold text-cyan-600 mb-2">100%</h3>
          <p className="text-gray-700 font-medium">Explainable</p>
        </div>
      </div>

      {/* CTA Section */}
      <div className="card medical-gradient text-white text-center">
        <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
        <p className="text-lg mb-6 text-blue-100">
          Upload your chest X-ray image and get instant AI-powered analysis
        </p>
        <Link to="/upload">
          <button className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-blue-50 transition-all duration-300">
            Start Analysis Now
          </button>
        </Link>
      </div>
    </div>
  )
}

export default HomePage
