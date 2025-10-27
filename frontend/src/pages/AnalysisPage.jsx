import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, Target, Activity, Award } from 'lucide-react'

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
    { name: 'Normal', value: 475, color: '#3b82f6' },
    { name: 'Pneumonia', value: 525, color: '#ef4444' },
  ]

  const COLORS = ['#3b82f6', '#ef4444']

  return (
    <div className="max-w-7xl mx-auto fade-in">
      <h1 className="text-4xl font-bold text-gray-800 mb-2 text-center">
        Model Performance Analysis
      </h1>
      <p className="text-gray-600 mb-8 text-center">
        Comprehensive evaluation metrics and visualizations of the pneumonia detection model
      </p>

      {/* Key Metrics Cards */}
      <div className="grid md:grid-cols-4 gap-6 mb-8">
        <div className="card bg-gradient-to-br from-blue-50 to-blue-100">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-700 font-semibold">Accuracy</h3>
            <Target className="w-6 h-6 text-blue-600" />
          </div>
          <p className="text-3xl font-bold text-blue-600">96.2%</p>
          <p className="text-sm text-gray-600 mt-1">Overall performance</p>
        </div>

        <div className="card bg-gradient-to-br from-green-50 to-green-100">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-700 font-semibold">Precision</h3>
            <Award className="w-6 h-6 text-green-600" />
          </div>
          <p className="text-3xl font-bold text-green-600">94.8%</p>
          <p className="text-sm text-gray-600 mt-1">Positive predictive value</p>
        </div>

        <div className="card bg-gradient-to-br from-purple-50 to-purple-100">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-700 font-semibold">Recall</h3>
            <Activity className="w-6 h-6 text-purple-600" />
          </div>
          <p className="text-3xl font-bold text-purple-600">95.5%</p>
          <p className="text-sm text-gray-600 mt-1">Sensitivity</p>
        </div>

        <div className="card bg-gradient-to-br from-orange-50 to-orange-100">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-gray-700 font-semibold">F1-Score</h3>
            <TrendingUp className="w-6 h-6 text-orange-600" />
          </div>
          <p className="text-3xl font-bold text-orange-600">95.1%</p>
          <p className="text-sm text-gray-600 mt-1">Harmonic mean</p>
        </div>
      </div>

      {/* Training History Charts */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Accuracy Chart */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Training & Validation Accuracy</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="training" stroke="#3b82f6" strokeWidth={2} name="Training" />
              <Line type="monotone" dataKey="validation" stroke="#10b981" strokeWidth={2} name="Validation" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Loss Chart */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Training & Validation Loss</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="training" stroke="#ef4444" strokeWidth={2} name="Training" />
              <Line type="monotone" dataKey="validation" stroke="#f59e0b" strokeWidth={2} name="Validation" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metrics and Distribution */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Metrics Bar Chart */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Performance Metrics</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Class Distribution */}
        <div className="card">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Test Set Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={classDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {classDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix */}
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Confusion Matrix</h2>
        <p className="text-gray-600 mb-6">
          Visual representation of model predictions vs actual labels
        </p>
        
        <div className="grid grid-cols-2 gap-4 max-w-2xl mx-auto">
          <div className="bg-green-100 border-2 border-green-300 rounded-lg p-6 text-center">
            <p className="text-sm text-gray-600 mb-2">True Negative</p>
            <p className="text-4xl font-bold text-green-700">450</p>
            <p className="text-xs text-gray-500 mt-2">Correctly predicted Normal</p>
          </div>
          
          <div className="bg-red-100 border-2 border-red-300 rounded-lg p-6 text-center">
            <p className="text-sm text-gray-600 mb-2">False Positive</p>
            <p className="text-4xl font-bold text-red-700">25</p>
            <p className="text-xs text-gray-500 mt-2">Normal predicted as Pneumonia</p>
          </div>
          
          <div className="bg-red-100 border-2 border-red-300 rounded-lg p-6 text-center">
            <p className="text-sm text-gray-600 mb-2">False Negative</p>
            <p className="text-4xl font-bold text-red-700">30</p>
            <p className="text-xs text-gray-500 mt-2">Pneumonia predicted as Normal</p>
          </div>
          
          <div className="bg-green-100 border-2 border-green-300 rounded-lg p-6 text-center">
            <p className="text-sm text-gray-600 mb-2">True Positive</p>
            <p className="text-4xl font-bold text-green-700">495</p>
            <p className="text-xs text-gray-500 mt-2">Correctly predicted Pneumonia</p>
          </div>
        </div>

        <div className="mt-6 bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-800 mb-2">Key Insights:</h3>
          <ul className="space-y-1 text-sm text-gray-700">
            <li>• <strong>Specificity:</strong> 94.7% (450/475) - Ability to correctly identify normal cases</li>
            <li>• <strong>Sensitivity:</strong> 94.3% (495/525) - Ability to correctly identify pneumonia cases</li>
            <li>• <strong>False Positive Rate:</strong> 5.3% - Low rate of false alarms</li>
            <li>• <strong>False Negative Rate:</strong> 5.7% - Low rate of missed pneumonia cases</li>
          </ul>
        </div>
      </div>

      {/* Model Information */}
      <div className="card mt-8 bg-gradient-to-br from-gray-50 to-gray-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Model Architecture</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Base Model</h3>
            <p className="text-gray-600">ResNet50 (Transfer Learning)</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Input Size</h3>
            <p className="text-gray-600">224 × 224 × 3</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Training Epochs</h3>
            <p className="text-gray-600">30 epochs</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Optimizer</h3>
            <p className="text-gray-600">Adam (lr=0.0001)</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Loss Function</h3>
            <p className="text-gray-600">Binary Cross-Entropy</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-700 mb-2">Data Augmentation</h3>
            <p className="text-gray-600">Rotation, Flip, Zoom</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisPage
