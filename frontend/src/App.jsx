import React, { useState, useRef } from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import { Activity, Upload, BarChart3, Info } from 'lucide-react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { ReactLenis } from '@studio-freight/react-lenis'
import HomePage from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import AnalysisPage from './pages/AnalysisPage'
import AboutPage from './pages/AboutPage'
import Cursor from './components/Cursor'
import GridBackground from './components/GridBackground'

import NeuralField from './components/NeuralField'

import ScrollToTop from './components/ScrollToTop'

function App() {
  const [navActive, setNavActive] = useState('home')
  const containerRef = useRef(null)
  const { scrollY } = useScroll()

  const y1 = useTransform(scrollY, [0, 1000], [0, 200])
  const y2 = useTransform(scrollY, [0, 1000], [0, -150])
  const rotate = useTransform(scrollY, [0, 1000], [0, 20])

  return (
    <ReactLenis root>
      <Router>
        <ScrollToTop />
        <Cursor />
        <div className="min-h-screen relative bg-[#09090b] text-slate-200 selection:bg-cyan-500 selection:text-white font-sans">
          <NeuralField />
          {/* Neural Grid Background */}
          <GridBackground />

          {/* Floating Navbar */}
          <nav className="fixed top-6 left-1/2 transform -translate-x-1/2 z-50 w-[90%] max-w-5xl">
            <div className="glass-dark rounded-full px-6 py-3 flex items-center justify-between">
              {/* Logo */}
              <Link to="/" className="flex items-center space-x-3 group" onClick={() => setNavActive('home')}>
                <div className="relative">
                  <div className="absolute inset-0 bg-cyan-500 blur-md opacity-50 group-hover:opacity-100 transition-opacity" />
                  <Activity className="w-6 h-6 text-cyan-400 relative z-10" />
                </div>
                <span className="text-white font-bold tracking-wide font-mono">PNEUMO<span className="text-cyan-400">VISION</span></span>
              </Link>

              {/* Links */}
              <div className="hidden md:flex items-center space-x-1">
                {[
                  { path: '/', label: 'Home', icon: Activity, id: 'home' },
                  { path: '/upload', label: 'Analyze', icon: Upload, id: 'upload' },
                  { path: '/analysis', label: 'Metrics', icon: BarChart3, id: 'analysis' },
                  { path: '/about', label: 'About', icon: Info, id: 'about' },
                ].map((item) => (
                  <Link
                    key={item.id}
                    to={item.path}
                    onClick={() => setNavActive(item.id)}
                    className={`relative px-4 py-2 rounded-full transition-all duration-300 group ${navActive === item.id ? 'text-white' : 'text-slate-400 hover:text-white'
                      }`}
                  >
                    {navActive === item.id && (
                      <motion.div
                        layoutId="nav-pill"
                        className="absolute inset-0 bg-white/10 rounded-full"
                        transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                      />
                    )}
                    <span className="relative z-10 flex items-center space-x-2 text-sm font-medium">
                      <item.icon className="w-4 h-4" />
                      <span>{item.label}</span>
                    </span>
                  </Link>
                ))}
              </div>
            </div>
          </nav>

          {/* Main Content */}
          <main className="relative z-10 pt-32 pb-20 px-6">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/analysis" element={<AnalysisPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </main>

          {/* Footer */}
          <footer className="relative z-10 border-t border-white/10 bg-black/20 backdrop-blur-md mt-20">
            <div className="container mx-auto px-6 py-12">
              <div className="flex flex-col md:flex-row justify-between items-center">
                <div className="mb-4 md:mb-0 text-center md:text-left">
                  <p className="text-sm text-slate-400">
                    © 2024 NeuroScan AI. Advancing Medical Diagnostics.
                  </p>
                  <p className="text-xs text-slate-600 mt-1">
                    Powered by ResNet50 & Grad-CAM Visualization
                  </p>
                </div>
                <div className="flex space-x-8">
                  {['Privacy', 'Terms', 'Contact'].map((item) => (
                    <a key={item} href="#" className="text-sm text-slate-400 hover:text-cyan-400 transition-colors">
                      {item}
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </ReactLenis>
  )
}

export default App
