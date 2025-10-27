import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { Activity, Upload, BarChart3, Info } from 'lucide-react'
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import AnalysisPage from './pages/AnalysisPage'
import AboutPage from './pages/AboutPage'

function App() {
  const [navActive, setNavActive] = useState('home')

  return (
    <Router>
      <div className="min-h-screen">
        {/* Navigation Bar */}
        <nav className="medical-gradient shadow-lg sticky top-0 z-50">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              {/* Logo and Title */}
              <Link to="/" className="flex items-center space-x-3" onClick={() => setNavActive('home')}>
                <Activity className="w-8 h-8 text-white" />
                <div>
                  <h1 className="text-white text-2xl font-bold">Pneumonia Detection</h1>
                  <p className="text-blue-100 text-xs">AI-Powered Medical Imaging</p>
                </div>
              </Link>

              {/* Navigation Links */}
              <div className="flex space-x-6">
                <Link
                  to="/"
                  onClick={() => setNavActive('home')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                    navActive === 'home'
                      ? 'bg-white bg-opacity-20 text-white'
                      : 'text-blue-100 hover:bg-white hover:bg-opacity-10'
                  }`}
                >
                  <Activity className="w-5 h-5" />
                  <span className="font-medium">Home</span>
                </Link>

                <Link
                  to="/upload"
                  onClick={() => setNavActive('upload')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                    navActive === 'upload'
                      ? 'bg-white bg-opacity-20 text-white'
                      : 'text-blue-100 hover:bg-white hover:bg-opacity-10'
                  }`}
                >
                  <Upload className="w-5 h-5" />
                  <span className="font-medium">Upload</span>
                </Link>

                <Link
                  to="/analysis"
                  onClick={() => setNavActive('analysis')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                    navActive === 'analysis'
                      ? 'bg-white bg-opacity-20 text-white'
                      : 'text-blue-100 hover:bg-white hover:bg-opacity-10'
                  }`}
                >
                  <BarChart3 className="w-5 h-5" />
                  <span className="font-medium">Analysis</span>
                </Link>

                <Link
                  to="/about"
                  onClick={() => setNavActive('about')}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
                    navActive === 'about'
                      ? 'bg-white bg-opacity-20 text-white'
                      : 'text-blue-100 hover:bg-white hover:bg-opacity-10'
                  }`}
                >
                  <Info className="w-5 h-5" />
                  <span className="font-medium">About</span>
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="container mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-gray-800 text-white mt-16">
          <div className="container mx-auto px-6 py-8">
            <div className="flex flex-col md:flex-row justify-between items-center">
              <div className="mb-4 md:mb-0">
                <p className="text-sm">
                  © 2024 Pneumonia Detection System. All rights reserved.
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  Powered by Deep Learning & Grad-CAM
                </p>
              </div>
              <div className="flex space-x-6">
                <a href="#" className="text-sm hover:text-blue-400 transition-colors">
                  Privacy Policy
                </a>
                <a href="#" className="text-sm hover:text-blue-400 transition-colors">
                  Terms of Service
                </a>
                <a href="#" className="text-sm hover:text-blue-400 transition-colors">
                  Contact
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  )
}

export default App
