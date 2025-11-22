import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, AreaChart, Area } from 'recharts'
import { TrendingUp, Target, Activity, Award, BarChart2, PieChart as PieIcon, Zap } from 'lucide-react'
import { motion } from 'framer-motion'

function AnalysisPage() {
  // Sample data for visualizations (replace with actual model metrics)
  const accuracyData = [
    { epoch: 1, training: 0.75, validation: 0.73 },
    { epoch: 5, training: 0.85, validation: 0.82 },
    { epoch: 10, training: 0.91, validation: 0.88 },
    { epoch: 15, training: 0.94, validation: 0.91 },
    { epoch: 20, training: 0.96, validation: 0.93 },
    { epoch: 25, training: 0.97, validation: 0.95 },
    { epoch: 30, training: 0.98, validation: 0.96 },
  ]

  const lossData = [
    { epoch: 1, training: 0.65, validation: 0.68 },
    { epoch: 5, training: 0.42, validation: 0.45 },
    { epoch: 10, training: 0.28, validation: 0.32 },
    { epoch: 15, training: 0.18, validation: 0.24 },
    { epoch: 20, training: 0.12, validation: 0.19 },
    { epoch: 25, training: 0.08, validation: 0.15 },
    { epoch: 30, training: 0.06, validation: 0.12 },
  ]

  const confusionMatrix = [
    { name: 'True Normal', predicted: 450, color: '#10b981' },
    { name: 'False Pneumonia', predicted: 25, color: '#ef4444' },
    { name: 'False Normal', predicted: 30, color: '#ef4444' },
    { name: 'True Pneumonia', predicted: 495, color: '#10b981' },
  ]

  const metricsData = [
    { metric: 'Accuracy', value: 96.2 },
    { metric: 'Precision', value: 94.8 },
    { metric: 'Recall', value: 95.5 },
    { metric: 'F1-Score', value: 95.1 },
  ]

  const classDistribution = [
    { name: 'Normal', value: 475, color: '#06b6d4' },
    { name: 'Pneumonia', value: 525, color: '#ec4899' },
  ]

  const COLORS = ['#06b6d4', '#ec4899']

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-900 border border-cyan-500/30 p-3 rounded shadow-xl backdrop-blur-md">
          <p className="text-cyan-400 font-mono text-xs mb-2">{`EPOCH: ${label}`}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-xs font-mono" style={{ color: entry.color }}>
              {`${entry.name.toUpperCase()}: ${entry.value}`}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="max-w-7xl mx-auto pt-12 pb-20 px-6">
      <div className="text-center mb-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center space-x-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full px-4 py-1.5 mb-6 backdrop-blur-sm"
        >
          <BarChart2 className="w-4 h-4 text-cyan-400" />
          <span className="text-cyan-400 text-sm font-mono tracking-wider">SYSTEM ANALYTICS</span>
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-4xl md:text-5xl font-bold text-white mb-4 font-mono"
        >
          METRICS DASHBOARD
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-slate-400 max-w-2xl mx-auto"
        >
          Real-time performance monitoring and model evaluation metrics.
        </motion.p>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid md:grid-cols-4 gap-6 mb-8">
        {[
          { label: "ACCURACY", value: "96.2%", icon: Target, color: "text-cyan-400", border: "border-cyan-500/30", bg: "bg-cyan-500/5" },
          { label: "PRECISION", value: "94.8%", icon: Award, color: "text-purple-400", border: "border-purple-500/30", bg: "bg-purple-500/5" },
          { label: "RECALL", value: "95.5%", icon: Activity, color: "text-pink-400", border: "border-pink-500/30", bg: "bg-pink-500/5" },
          { label: "F1-SCORE", value: "95.1%", icon: TrendingUp, color: "text-yellow-400", border: "border-yellow-500/30", bg: "bg-yellow-500/5" }
        ].map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.1 }}
            className={`glass-dark p-6 rounded-xl border ${stat.border} ${stat.bg} relative overflow-hidden group`}
          >
            <div className="absolute top-0 right-0 p-4 opacity-20 group-hover:opacity-40 transition-opacity">
              <stat.icon className={`w-12 h-12 ${stat.color}`} />
            </div>
            <h3 className={`text-sm font-mono ${stat.color} mb-2`}>{stat.label}</h3>
            <p className="text-4xl font-bold text-white font-mono">{stat.value}</p>
            <div className="w-full bg-slate-800 h-1 mt-4 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: stat.value }}
                transition={{ duration: 1.5, ease: "easeOut" }}
                className={`h-full ${stat.color.replace('text-', 'bg-')}`}
              />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Training History Charts */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Accuracy Chart */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-dark p-6 rounded-2xl border border-white/10"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white font-mono flex items-center">
              <Zap className="w-5 h-5 mr-2 text-cyan-400" />
              LEARNING CURVE
            </h2>
            <div className="flex space-x-2 text-xs font-mono">
              <span className="text-cyan-400">● TRAINING</span>
              <span className="text-purple-400">● VALIDATION</span>
            </div>
          </div>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={accuracyData}>
                <defs>
                  <linearGradient id="colorTrain" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="epoch" stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} domain={[0.6, 1]} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="training" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#colorTrain)" />
                <Area type="monotone" dataKey="validation" stroke="#a855f7" strokeWidth={2} fillOpacity={1} fill="url(#colorVal)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Loss Chart */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
          className="glass-dark p-6 rounded-2xl border border-white/10"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white font-mono flex items-center">
              <Activity className="w-5 h-5 mr-2 text-pink-400" />
              LOSS CONVERGENCE
            </h2>
            <div className="flex space-x-2 text-xs font-mono">
              <span className="text-pink-400">● TRAINING</span>
              <span className="text-yellow-400">● VALIDATION</span>
            </div>
          </div>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="epoch" stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="training" stroke="#ec4899" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="validation" stroke="#eab308" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>
      </div>

      {/* Metrics and Distribution */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Metrics Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-dark p-6 rounded-2xl border border-white/10"
        >
          <h2 className="text-xl font-bold text-white font-mono mb-6 flex items-center">
            <BarChart2 className="w-5 h-5 mr-2 text-cyan-400" />
            PERFORMANCE BREAKDOWN
          </h2>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} />
                <YAxis dataKey="metric" type="category" stroke="#94a3b8" tick={{ fontSize: 12, fontFamily: 'monospace' }} width={100} />
                <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} content={<CustomTooltip />} />
                <Bar dataKey="value" fill="#06b6d4" radius={[0, 4, 4, 0]} barSize={20}>
                  {metricsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#06b6d4' : '#8b5cf6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Class Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-dark p-6 rounded-2xl border border-white/10"
        >
          <h2 className="text-xl font-bold text-white font-mono mb-6 flex items-center">
            <PieIcon className="w-5 h-5 mr-2 text-pink-400" />
            DATASET BALANCE
          </h2>
          <div className="h-[300px] w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={classDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {classDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="rgba(0,0,0,0)" />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend verticalAlign="bottom" height={36} iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
            {/* Center Text */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none pb-8">
              <div className="text-center">
                <span className="text-2xl font-bold text-white font-mono">1000</span>
                <span className="block text-xs text-slate-500 font-mono">SAMPLES</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Confusion Matrix */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="glass-dark p-8 rounded-2xl border border-white/10"
      >
        <h2 className="text-xl font-bold text-white font-mono mb-8 text-center">CONFUSION MATRIX</h2>

        <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-6 text-center hover:bg-green-500/20 transition-colors">
            <p className="text-xs font-mono text-green-400 mb-2">TRUE NEGATIVE</p>
            <p className="text-4xl font-bold text-white font-mono">450</p>
            <p className="text-[10px] text-slate-400 mt-2 uppercase tracking-wider">Correctly Identified Normal</p>
          </div>

          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 text-center hover:bg-red-500/20 transition-colors">
            <p className="text-xs font-mono text-red-400 mb-2">FALSE POSITIVE</p>
            <p className="text-4xl font-bold text-white font-mono">25</p>
            <p className="text-[10px] text-slate-400 mt-2 uppercase tracking-wider">Type I Error</p>
          </div>

          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 text-center hover:bg-red-500/20 transition-colors">
            <p className="text-xs font-mono text-red-400 mb-2">FALSE NEGATIVE</p>
            <p className="text-4xl font-bold text-white font-mono">30</p>
            <p className="text-[10px] text-slate-400 mt-2 uppercase tracking-wider">Type II Error</p>
          </div>

          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-6 text-center hover:bg-green-500/20 transition-colors">
            <p className="text-xs font-mono text-green-400 mb-2">TRUE POSITIVE</p>
            <p className="text-4xl font-bold text-white font-mono">495</p>
            <p className="text-[10px] text-slate-400 mt-2 uppercase tracking-wider">Correctly Identified Pneumonia</p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default AnalysisPage
