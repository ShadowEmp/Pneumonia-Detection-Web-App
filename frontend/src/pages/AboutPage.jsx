import React from 'react'
import { Brain, Eye, Zap, Shield, Users, BookOpen, Github, Mail, Activity, Database, Server, Code, Layers } from 'lucide-react'
import { motion } from 'framer-motion'

function AboutPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="max-w-7xl mx-auto pt-12 pb-20 px-6"
    >
      <div className="text-center mb-16">
        <motion.div variants={itemVariants} className="inline-flex items-center space-x-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full px-4 py-1.5 mb-6 backdrop-blur-sm">
          <Server className="w-4 h-4 text-cyan-400" />
          <span className="text-cyan-400 text-sm font-mono tracking-wider">SYSTEM DOCUMENTATION</span>
        </motion.div>
        <motion.h1 variants={itemVariants} className="text-4xl md:text-5xl font-bold text-white mb-6 font-mono">
          PROJECT SPECIFICATIONS
        </motion.h1>
        <motion.p variants={itemVariants} className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
          Bridging the gap between artificial intelligence and medical diagnostics through transparent, explainable deep learning architectures.
        </motion.p>
      </div>

      {/* Project Overview */}
      <motion.div variants={itemVariants} className="glass-dark rounded-3xl p-1 border border-white/10 relative overflow-hidden mb-16">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl -mr-32 -mt-32 pointer-events-none" />

        <div className="bg-slate-900/50 p-10 rounded-[20px] relative z-10">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <div className="flex items-center space-x-2 mb-6">
                <div className="w-1 h-6 bg-cyan-500 rounded-full" />
                <h2 className="text-2xl font-bold text-white font-mono">MISSION LOG</h2>
              </div>
              <div className="space-y-6 font-mono text-sm text-slate-300">
                <p className="leading-relaxed border-l-2 border-slate-700 pl-4">
                  <span className="text-cyan-400 block mb-2">&gt;&gt; PROBLEM_STATEMENT</span>
                  Pneumonia remains a leading cause of death worldwide. Early and accurate diagnosis is crucial for effective treatment protocols.
                </p>
                <p className="leading-relaxed border-l-2 border-slate-700 pl-4">
                  <span className="text-cyan-400 block mb-2">&gt;&gt; SOLUTION_ARCHITECTURE</span>
                  Leveraging ResNet50 transfer learning to assist radiologists with instant, second-opinion screenings. Integrated Grad-CAM visualization ensures diagnostic transparency.
                </p>
              </div>
            </div>

            <div className="relative h-64 perspective-1000">
              <motion.div
                animate={{ rotateY: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 flex items-center justify-center transform preserve-3d"
              >
                <div className="relative w-48 h-48">
                  {/* Rotating Rings */}
                  <div className="absolute inset-0 border-2 border-cyan-500/30 rounded-full transform rotate-x-45" />
                  <div className="absolute inset-0 border-2 border-purple-500/30 rounded-full transform -rotate-x-45" />
                  <div className="absolute inset-0 border-2 border-white/10 rounded-full" />

                  {/* Center Core */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Brain className="w-24 h-24 text-cyan-400 drop-shadow-[0_0_15px_rgba(6,182,212,0.5)]" />
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Tech Stack Grid */}
      <motion.div variants={itemVariants} className="mb-16">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-2xl font-bold text-white font-mono flex items-center">
            <Code className="w-6 h-6 mr-3 text-cyan-400" />
            TECHNOLOGY STACK
          </h2>
          <div className="h-px flex-1 bg-slate-800 ml-6" />
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { icon: Brain, title: "TENSORFLOW", desc: "Deep Learning Core", color: "text-orange-400", border: "hover:border-orange-500/50" },
            { icon: Layers, title: "RESNET50", desc: "CNN Architecture", color: "text-blue-400", border: "hover:border-blue-500/50" },
            { icon: Eye, title: "GRAD-CAM", desc: "Visual Explainability", color: "text-green-400", border: "hover:border-green-500/50" },
            { icon: Zap, title: "REACT + VITE", desc: "Frontend Engine", color: "text-cyan-400", border: "hover:border-cyan-500/50" }
          ].map((tech, idx) => (
            <motion.div
              key={idx}
              whileHover={{ y: -5, scale: 1.02 }}
              className={`glass-dark p-6 rounded-xl border border-white/5 ${tech.border} transition-all group cursor-default`}
            >
              <div className="w-12 h-12 bg-slate-800 rounded-lg flex items-center justify-center mb-4 group-hover:bg-white/5 transition-colors">
                <tech.icon className={`w-6 h-6 ${tech.color}`} />
              </div>
              <h3 className="font-bold text-white font-mono mb-1">{tech.title}</h3>
              <p className="text-xs text-slate-400 font-mono uppercase tracking-wide">{tech.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Dataset Info */}
      <motion.div variants={itemVariants} className="grid md:grid-cols-2 gap-8 mb-16">
        <div className="glass-dark p-1 rounded-3xl border border-white/10">
          <div className="bg-slate-900/50 p-8 rounded-[20px] h-full">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center font-mono">
              <Database className="w-5 h-5 text-cyan-400 mr-3" />
              DATASET METRICS
            </h3>
            <div className="space-y-4">
              {[
                { label: "SOURCE", value: "Chest X-Ray Images (Pneumonia)" },
                { label: "TRAINING SET", value: "90,000 Images" },
                { label: "TEST SET", value: "10,000 Images" },
                { label: "CLASSES", value: "Normal vs. Pneumonia" }
              ].map((item, idx) => (
                <div key={idx} className="flex justify-between items-center border-b border-white/5 pb-3 last:border-0">
                  <span className="text-xs font-mono text-slate-500">{item.label}</span>
                  <span className="text-sm font-mono text-white">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="glass-dark p-1 rounded-3xl border border-yellow-500/30 relative overflow-hidden">
          <div className="absolute -right-10 -top-10 w-40 h-40 bg-yellow-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="bg-slate-900/50 p-8 rounded-[20px] h-full relative z-10">
            <h3 className="text-xl font-bold text-white mb-6 flex items-center font-mono">
              <Shield className="w-5 h-5 text-yellow-400 mr-3" />
              LEGAL DISCLAIMER
            </h3>
            <p className="text-slate-300 text-sm leading-relaxed mb-6 font-mono">
              This system is engineered as a <strong className="text-yellow-400">diagnostic assistance tool</strong>. It is NOT intended to replace professional medical judgment or laboratory diagnosis.
            </p>
            <div className="p-4 bg-yellow-500/5 rounded-xl border border-yellow-500/20">
              <p className="text-xs text-yellow-200/80 font-mono">
                &gt;&gt; WARNING: All predictions must be verified by qualified healthcare personnel.
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Contact */}
      <motion.div variants={itemVariants} className="text-center">
        <h2 className="text-2xl font-bold text-white mb-8 font-mono">INITIALIZE CONTACT</h2>
        <div className="flex justify-center gap-4">
          <a href="#" className="group flex items-center space-x-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 border border-slate-600 hover:border-white rounded-full text-white transition-all">
            <Github className="w-4 h-4 group-hover:scale-110 transition-transform" />
            <span className="font-mono text-sm">GITHUB REPO</span>
          </a>
          <a href="#" className="group flex items-center space-x-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 border border-cyan-400 hover:border-white rounded-full text-white transition-all">
            <Mail className="w-4 h-4 group-hover:scale-110 transition-transform" />
            <span className="font-mono text-sm">CONTACT TEAM</span>
          </a>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default AboutPage
