import React, { useRef } from 'react'
import { Link } from 'react-router-dom'
import { Upload, Activity, Zap, Shield, ArrowRight, Cpu, Scan, FileSearch, Layers, Database, Network } from 'lucide-react'
import { motion, useScroll, useTransform } from 'framer-motion'

function HomePage() {
  // Refs for sections
  const heroRef = useRef(null)
  const featuresRef = useRef(null)
  const systemRef = useRef(null)
  const statsRef = useRef(null)

  // Hero Scroll Transforms
  const { scrollYProgress: heroProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"]
  })
  const heroRotateX = useTransform(heroProgress, [0, 1], [0, 45])
  const heroOpacity = useTransform(heroProgress, [0, 0.8], [1, 0])
  const heroScale = useTransform(heroProgress, [0, 1], [1, 0.8])

  // Features Scroll Transforms
  const { scrollYProgress: featuresProgress } = useScroll({
    target: featuresRef,
    offset: ["start end", "end start"]
  })
  const featuresRotateX = useTransform(featuresProgress, [0.1, 0.5], [45, 0])
  const featuresOpacity = useTransform(featuresProgress, [0, 0.3], [0, 1])

  // System Architecture Scroll Transforms
  const { scrollYProgress: systemProgress } = useScroll({
    target: systemRef,
    offset: ["start start", "end end"]
  })

  // Phase 1: Explode (0-0.5) | Phase 2: Reassemble (0.5-1.0)
  const systemRotateX = useTransform(systemProgress, [0, 0.5, 1], [45, 70, 0]) // Twist then flatten
  const systemRotateZ = useTransform(systemProgress, [0, 0.5, 1], [-45, 45, 0]) // Twist then straighten
  const systemScale = useTransform(systemProgress, [0, 0.5, 1], [0.8, 1.1, 0.7]) // Pulse then shrink to fit width
  const systemY = useTransform(systemProgress, [0, 0.5, 1], [0, 0, 0]) // Keep vertical center
  // Removed systemX - we will move individual layers symmetrically

  // Layer expansions (Chaotic Deconstruction -> Horizontal Line)
  // Base (Input) -> Move to X=-540
  const baseX = useTransform(systemProgress, [0, 0.5, 1], [0, 0, -540])

  // Layer 1: ResNet -> Move to X=-180
  const layer1Z = useTransform(systemProgress, [0, 0.5, 1], [0, 200, 0])
  const layer1Y = useTransform(systemProgress, [0, 0.5, 1], [0, -100, 0])
  const layer1X = useTransform(systemProgress, [0, 0.5, 1], [0, -150, -180])
  const layer1Rotate = useTransform(systemProgress, [0, 0.5, 1], [0, -25, 0])

  // Layer 2: Feature Maps -> Move to X=180
  const layer2Z = useTransform(systemProgress, [0, 0.5, 1], [0, 400, 0])
  const layer2Y = useTransform(systemProgress, [0, 0.5, 1], [0, -200, 0])
  const layer2X = useTransform(systemProgress, [0, 0.5, 1], [0, 0, 180])
  const layer2Rotate = useTransform(systemProgress, [0, 0.5, 1], [0, 10, 0])

  // Layer 3: Diagnosis -> Move to X=540
  const layer3Z = useTransform(systemProgress, [0, 0.5, 1], [0, 600, 0])
  const layer3Y = useTransform(systemProgress, [0, 0.5, 1], [0, -300, 0])
  const layer3X = useTransform(systemProgress, [0, 0.5, 1], [0, 150, 540])
  const layer3Rotate = useTransform(systemProgress, [0, 0.5, 1], [0, 25, 0])

  // Description Opacity - Visible during explosion and reassembly
  const descOpacity = useTransform(systemProgress, [0.2, 0.5], [0, 1])

  // Stats Scroll Transforms
  const { scrollYProgress: statsProgress } = useScroll({
    target: statsRef,
    offset: ["start end", "end start"]
  })
  const statsRotateX = useTransform(statsProgress, [0, 1], [45, -45])
  const statsOpacity = useTransform(statsProgress, [0, 0.5, 1], [0, 1, 0])

  const staggerContainer = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const fadeInUp = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } }
  }

  return (
    <div className="relative z-10 perspective-2000">
      {/* Hero Section */}
      <section ref={heroRef} className="min-h-screen flex items-center justify-center relative pt-20 preserve-3d">
        <motion.div
          style={{ rotateX: heroRotateX, opacity: heroOpacity, scale: heroScale }}
          className="container mx-auto px-6 relative z-10 text-center origin-center"
        >
          <motion.div
            variants={staggerContainer}
            initial="hidden"
            animate="show"
            className="max-w-4xl mx-auto"
          >
            <motion.div variants={fadeInUp} className="inline-flex items-center space-x-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full px-4 py-1.5 mb-8 backdrop-blur-sm">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
              </span>
              <span className="text-cyan-400 text-sm font-mono tracking-wider">SYSTEM ONLINE v2.0</span>
            </motion.div>

            <motion.h1 variants={fadeInUp} className="text-6xl md:text-8xl font-bold text-white mb-8 tracking-tight leading-tight">
              <span className="block text-transparent bg-clip-text bg-gradient-to-b from-white to-slate-500">PNEUMONIA</span>
              <span className="block text-cyan-400 font-mono">DETECTION</span>
            </motion.h1>

            <motion.p variants={fadeInUp} className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              Advanced AI-powered pneumonia detection system utilizing ResNet50 architecture for instant, high-precision X-ray analysis.
            </motion.p>

            <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row items-center justify-center gap-6">
              <Link
                to="/upload"
                className="group relative px-8 py-4 bg-cyan-500 text-black font-bold text-lg rounded-none overflow-hidden transition-all hover:bg-cyan-400"
              >
                <div className="absolute inset-0 w-full h-full bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                <span className="relative flex items-center">
                  INITIALIZE SCAN <Scan className="ml-2 w-5 h-5" />
                </span>
              </Link>
              <Link
                to="/about"
                className="group px-8 py-4 bg-transparent border border-slate-700 text-white font-mono text-lg hover:border-cyan-500/50 hover:text-cyan-400 transition-all"
              >
                VIEW SPECS
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section ref={featuresRef} className="py-32 relative perspective-1000">
        <motion.div
          style={{ rotateX: featuresRotateX, opacity: featuresOpacity }}
          className="container mx-auto px-6 origin-bottom"
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-4xl font-bold text-white mb-4 font-mono">CORE MODULES</h2>
            <div className="h-1 w-20 bg-cyan-500 mx-auto" />
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Cpu,
                title: "Deep Learning Core",
                desc: "Powered by ResNet50 CNN architecture trained on 5,000+ medical images."
              },
              {
                icon: FileSearch,
                title: "Visual Explanation",
                desc: "Grad-CAM technology highlights the exact regions of interest in X-rays."
              },
              {
                icon: Zap,
                title: "Real-time Analysis",
                desc: "Instant processing pipeline delivers results in under 2 seconds."
              }
            ].map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1 }}
                whileHover={{ y: -5 }}
                className="group relative p-8 bg-slate-900/50 border border-slate-800 hover:border-cyan-500/50 transition-all duration-300"
              >
                <div className="absolute inset-0 bg-cyan-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <feature.icon className="w-12 h-12 text-cyan-500 mb-6 group-hover:scale-110 transition-transform duration-300" />
                <h3 className="text-xl font-bold text-white mb-4 font-mono">{feature.title}</h3>
                <p className="text-slate-400 leading-relaxed">{feature.desc}</p>

                {/* Corner Accents */}
                <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan-500/30 group-hover:border-cyan-500 transition-colors" />
                <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan-500/30 group-hover:border-cyan-500 transition-colors" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* System Architecture - Scroll Rotation Section */}
      <section ref={systemRef} className="h-[300vh] relative perspective-2000">
        <div className="sticky top-0 h-screen flex flex-col items-center justify-start pt-32">
          <div className="container mx-auto px-6 relative z-10">
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-white mb-4 font-mono">SYSTEM ARCHITECTURE</h2>
              <p className="text-slate-400">Scroll to deconstruct the neural network layers</p>
            </div>
          </div>

          <div className="w-full h-[600px] flex items-center justify-center relative preserve-3d">
            <motion.div
              style={{ rotateX: systemRotateX, rotateZ: systemRotateZ, scale: systemScale, y: systemY }}
              className="relative w-80 h-80 preserve-3d"
            >
              {/* Base Plate (Input) */}
              <motion.div
                style={{ x: baseX }}
                className="absolute inset-0 bg-slate-900/80 border border-slate-700 rounded-xl flex items-center justify-center shadow-2xl backdrop-blur-sm"
              >
                <div className="text-center p-4">
                  <FileSearch className="w-16 h-16 text-slate-500 mx-auto mb-2" />
                  <span className="font-mono text-lg font-bold text-slate-300 block">INPUT DATA</span>
                  <motion.p style={{ opacity: descOpacity }} className="text-xs text-slate-400 mt-2">
                    Raw X-ray image pre-processed for analysis.
                  </motion.p>
                </div>
              </motion.div>

              {/* Layer 1 (Processing) */}
              <motion.div
                style={{ z: layer1Z, y: layer1Y, x: layer1X, rotateZ: layer1Rotate }}
                className="absolute inset-0 bg-cyan-900/40 border border-cyan-500/30 rounded-xl flex items-center justify-center shadow-[0_0_30px_rgba(6,182,212,0.2)] backdrop-blur-sm"
              >
                <div className="text-center p-4">
                  <Layers className="w-12 h-12 text-cyan-400 mx-auto mb-2" />
                  <span className="font-mono text-sm text-cyan-300 block">RESNET50 LAYERS</span>
                  <motion.p style={{ opacity: descOpacity }} className="text-xs text-cyan-200/70 mt-2">
                    Deep convolutional blocks extracting hierarchical features.
                  </motion.p>
                </div>
              </motion.div>

              {/* Layer 2 (Feature Extraction) */}
              <motion.div
                style={{ z: layer2Z, y: layer2Y, x: layer2X, rotateZ: layer2Rotate }}
                className="absolute inset-0 bg-blue-900/40 border border-blue-500/30 rounded-xl flex items-center justify-center shadow-[0_0_30px_rgba(59,130,246,0.2)] backdrop-blur-sm"
              >
                <div className="text-center p-4">
                  <Network className="w-12 h-12 text-blue-400 mx-auto mb-2" />
                  <span className="font-mono text-sm text-blue-300 block">FEATURE MAPS</span>
                  <motion.p style={{ opacity: descOpacity }} className="text-xs text-blue-200/70 mt-2">
                    High-level activation maps identifying pathological patterns.
                  </motion.p>
                </div>
              </motion.div>

              {/* Layer 3 (Output) */}
              <motion.div
                style={{ z: layer3Z, y: layer3Y, x: layer3X, rotateZ: layer3Rotate }}
                className="absolute inset-0 bg-gradient-to-br from-pink-500/20 to-purple-500/20 border border-pink-500/30 rounded-xl flex items-center justify-center shadow-[0_0_30px_rgba(236,72,153,0.3)] backdrop-blur-sm"
              >
                <div className="text-center p-4">
                  <Activity className="w-16 h-16 text-pink-400 mx-auto mb-2 animate-pulse" />
                  <span className="font-mono text-lg font-bold text-pink-300 block">DIAGNOSIS</span>
                  <motion.p style={{ opacity: descOpacity }} className="text-xs text-pink-200/70 mt-2">
                    Final classification probability for Pneumonia presence.
                  </motion.p>
                </div>
              </motion.div>

              {/* Connecting Lines (Visual only) */}
              <div className="absolute inset-0 border border-white/5 rounded-xl transform translate-z-[150px] pointer-events-none" />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      < section ref={statsRef} className="mt-32 py-32 border-t border-slate-800 bg-slate-900/30 backdrop-blur-sm perspective-1000 relative z-20" >
        <motion.div
          style={{ rotateX: statsRotateX, opacity: statsOpacity }}
          className="container mx-auto px-6 origin-center"
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-4xl font-bold text-white mb-4 font-mono">PERFORMANCE METRICS</h2>
            <div className="h-1 w-20 bg-cyan-500 mx-auto" />
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-12 text-center">
            {[
              { label: "ACCURACY", value: "96.4%" },
              { label: "PRECISION", value: "98.2%" },
              { label: "RECALL", value: "97.1%" },
              { label: "LATENCY", value: "<2s" }
            ].map((stat, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: idx * 0.1 }}
              >
                <div className="text-4xl md:text-5xl font-bold text-white mb-2 font-mono">{stat.value}</div>
                <div className="text-sm text-cyan-500 tracking-widest font-mono">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section >
    </div >
  )
}

export default HomePage
