import React, { useState, useCallback, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Upload, X, Loader, CheckCircle, AlertCircle, Download, AlertTriangle, Scan, FileSearch, Activity, Cpu, Stethoscope, User, Maximize2, ZoomIn, RefreshCw, ArrowRight } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { jsPDF } from 'jspdf'

function UploadPage() {
  // State: 'idle' | 'ready' | 'scanning' | 'result'
  const [viewState, setViewState] = useState('idle')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [selectedImage, setSelectedImage] = useState(null)
  const fileInputRef = useRef(null)

  // Scanning simulation state
  const [scanProgress, setScanProgress] = useState(0)

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
      setViewState('ready')
    }
  }, [])

  const handleFileInputChange = (e) => {
    const files = e.target.files
    if (files && files.length > 0) {
      onDrop([files[0]])
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1,
    multiple: false,
    noClick: true, // We handle clicks manually
    noKeyboard: true
  })

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setViewState('idle')
    setScanProgress(0)
  }

  const handlePredict = async () => {
    if (!file) return

    setViewState('scanning')
    setError(null)
    setScanProgress(0)

    // Simulate scanning progress
    const progressInterval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 90) return prev
        return prev + 5
      })
    }, 100)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Use relative path to leverage Vite proxy
      const response = await axios.post('/api/predict-with-gradcam', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      clearInterval(progressInterval)
      setScanProgress(100)

      // Small delay to show 100%
      setTimeout(() => {
        setResult(response.data)
        setViewState('result')
      }, 500)

    } catch (err) {
      clearInterval(progressInterval)
      setError(err.response?.data?.error || 'An error occurred during prediction')
      setViewState('ready') // Go back to ready state on error
    }
  }

  const downloadImage = (imageData, filename) => {
    const link = document.createElement('a')
    link.href = imageData
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const generateReport = () => {
    const doc = new jsPDF()
    const pageWidth = doc.internal.pageSize.getWidth()
    const pageHeight = doc.internal.pageSize.getHeight()
    const margin = 20

    // --- HEADER ---
    doc.setFillColor(6, 182, 212) // Cyan-500
    doc.rect(0, 0, pageWidth, 40, 'F')

    doc.setTextColor(255, 255, 255)
    doc.setFontSize(24)
    doc.setFont('helvetica', 'bold')
    doc.text("PNEUMOVISION", margin, 20)

    doc.setFontSize(12)
    doc.setFont('helvetica', 'normal')
    doc.text("AI-POWERED DIAGNOSTIC REPORT", margin, 30)

    doc.setFontSize(10)
    doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - margin - 60, 20)
    doc.text(`ID: ${Math.random().toString(36).substr(2, 9).toUpperCase()}`, pageWidth - margin - 60, 30)

    // --- DIAGNOSIS RESULT ---
    let yPos = 60
    doc.setTextColor(0, 0, 0)
    doc.setFontSize(16)
    doc.setFont('helvetica', 'bold')
    doc.text("DIAGNOSTIC RESULT", margin, yPos)

    yPos += 15
    const isPneumonia = result.prediction.is_pneumonia
    const color = isPneumonia ? [239, 68, 68] : [34, 197, 94] // Red or Green

    doc.setFillColor(...color)
    doc.roundedRect(margin, yPos - 10, pageWidth - (margin * 2), 25, 3, 3, 'F')

    doc.setTextColor(255, 255, 255)
    doc.setFontSize(18)
    doc.text(isPneumonia ? "PNEUMONIA DETECTED" : "NORMAL / NO OPACITY", margin + 10, yPos + 7)

    doc.setFontSize(12)
    doc.text(`Confidence: ${(result.prediction.confidence * 100).toFixed(1)}%`, pageWidth - margin - 50, yPos + 7)

    // --- VISUAL EVIDENCE ---
    yPos += 40
    doc.setTextColor(0, 0, 0)
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text("VISUAL EVIDENCE", margin, yPos)

    yPos += 10
    const imgWidth = (pageWidth - (margin * 3)) / 2
    const imgHeight = imgWidth * 0.75

    try {
      doc.addImage(result.gradcam.original, 'JPEG', margin, yPos, imgWidth, imgHeight)
      doc.addImage(result.gradcam.overlay, 'JPEG', margin + imgWidth + margin, yPos, imgWidth, imgHeight)

      doc.setFontSize(10)
      doc.setFont('helvetica', 'normal')
      doc.text("Original X-Ray", margin, yPos + imgHeight + 5)
      doc.text("AI Heatmap Analysis", margin + imgWidth + margin, yPos + imgHeight + 5)
    } catch (e) {
      console.error("Error adding images to PDF", e)
    }

    // --- DETAILED ANALYSIS ---
    yPos += imgHeight + 25
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text("DETAILED ANALYSIS", margin, yPos)

    yPos += 10
    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    const analysisText = isPneumonia
      ? `The AI model has detected patterns consistent with pneumonia with ${(result.prediction.confidence * 100).toFixed(1)}% confidence. The Grad-CAM heatmap (right image) highlights regions of significant activation in red/yellow, which typically correspond to lung consolidation, infiltrates, or other opacities associated with infection.`
      : `The AI model analyzed the chest X-ray and found no significant signs of pneumonia, with a confidence of ${(result.prediction.confidence * 100).toFixed(1)}% for a Normal diagnosis. The heatmap shows no concentrated regions of concern typically associated with pathological opacities.`

    const splitAnalysis = doc.splitTextToSize(analysisText, pageWidth - (margin * 2))
    doc.text(splitAnalysis, margin, yPos)
    yPos += (splitAnalysis.length * 5) + 10

    // --- NEW PAGE FOR ACTION PLAN & EDUCATION ---
    doc.addPage()
    yPos = 20 // Reset Y position for new page

    // --- ACTION PLAN ---
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text("COMPREHENSIVE ACTION PLAN", margin, yPos)
    yPos += 10

    // Patient Section
    doc.setFillColor(240, 249, 255) // Light Blue
    doc.rect(margin, yPos, (pageWidth - (margin * 3)) / 2, 85, 'F') // Increased height

    doc.setTextColor(6, 182, 212)
    doc.setFontSize(12)
    doc.text("FOR PATIENTS", margin + 5, yPos + 10)

    doc.setTextColor(0, 0, 0)
    doc.setFontSize(9)
    const patientSteps = isPneumonia
      ? [
        "1. IMMEDIATE: Schedule an appointment with a pulmonologist.",
        "2. MONITOR: Track fever (>100.4F), cough, and breathing.",
        "3. REST: Ensure adequate sleep and fluid intake.",
        "4. ISOLATE: Stay home if you suspect a contagious infection.",
        "5. EMERGENCY: Go to ER if lips turn blue or chest pain is severe."
      ]
      : [
        "1. MONITOR: Watch for any developing symptoms.",
        "2. HYGIENE: Wash hands frequently and avoid sick contacts.",
        "3. VACCINATION: Stay up to date with flu and pneumococcal vaccines.",
        "4. LIFESTYLE: Maintain a healthy diet and exercise routine.",
        "5. FOLLOW-UP: Consult a doctor if you feel unwell."
      ]

    let stepY = yPos + 20
    patientSteps.forEach(step => {
      const splitStep = doc.splitTextToSize(step, ((pageWidth - (margin * 3)) / 2) - 10)
      doc.text(splitStep, margin + 5, stepY)
      stepY += (splitStep.length * 4) + 4
    })

    // Doctor Section
    doc.setFillColor(240, 253, 244) // Light Green
    doc.rect(margin + ((pageWidth - (margin * 3)) / 2) + margin, yPos, (pageWidth - (margin * 3)) / 2, 85, 'F') // Increased height

    doc.setTextColor(22, 163, 74)
    doc.setFontSize(12)
    doc.text("FOR PHYSICIANS", margin + ((pageWidth - (margin * 3)) / 2) + margin + 5, yPos + 10)

    doc.setTextColor(0, 0, 0)
    doc.setFontSize(9)
    const doctorSteps = isPneumonia
      ? [
        "1. CLINICAL CORRELATION: Auscultate for crackles/wheezing.",
        "2. CONFIRMATION: Consider lateral X-ray or CT scan.",
        "3. LAB WORK: Order CBC, CRP, ESR to distinguish etiology.",
        "4. HISTORY: Review patient history for comorbidities.",
        "5. DIFFERENTIAL: Rule out pulmonary edema or malignancy."
      ]
      : [
        "1. REVIEW: Check for subtle opacities missed by AI.",
        "2. HISTORY: Evaluate risk factors (smoking, exposure).",
        "3. SYMPTOMS: If symptomatic, consider other etiologies.",
        "4. FOLLOW-UP: Schedule follow-up if symptoms worsen.",
        "5. DOCUMENTATION: Record findings in EMR."
      ]

    stepY = yPos + 20
    doctorSteps.forEach(step => {
      const splitStep = doc.splitTextToSize(step, ((pageWidth - (margin * 3)) / 2) - 10)
      doc.text(splitStep, margin + ((pageWidth - (margin * 3)) / 2) + margin + 5, stepY)
      stepY += (splitStep.length * 4) + 4
    })

    yPos += 100

    // --- UNDERSTANDING PNEUMONIA (EDUCATIONAL CONTENT) ---
    doc.setFontSize(14)
    doc.setFont('helvetica', 'bold')
    doc.text("UNDERSTANDING PNEUMONIA", margin, yPos)
    yPos += 10

    doc.setFontSize(10)
    doc.setFont('helvetica', 'normal')
    const definition = "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia."
    const splitDef = doc.splitTextToSize(definition, pageWidth - (margin * 2))
    doc.text(splitDef, margin, yPos)
    yPos += (splitDef.length * 5) + 10

    // Symptoms & Risk Factors Grid
    const colWidth = (pageWidth - (margin * 3)) / 2

    // Symptoms
    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.text("COMMON SYMPTOMS", margin, yPos)
    doc.setFontSize(9)
    doc.setFont('helvetica', 'normal')
    const symptoms = [
      "• Chest pain when you breathe or cough",
      "• Confusion or changes in mental awareness (in adults age 65 and older)",
      "• Cough, which may produce phlegm",
      "• Fatigue",
      "• Fever, sweating and shaking chills",
      "• Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)",
      "• Nausea, vomiting or diarrhea",
      "• Shortness of breath"
    ]
    let symY = yPos + 8
    symptoms.forEach(sym => {
      const splitSym = doc.splitTextToSize(sym, colWidth)
      doc.text(splitSym, margin, symY)
      const dim = doc.getTextDimensions(splitSym)
      symY += dim.h + 3
    })

    // Prevention
    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.text("PREVENTION & CARE", margin + colWidth + margin, yPos)
    doc.setFontSize(9)
    doc.setFont('helvetica', 'normal')
    const prevention = [
      "• Get vaccinated: Vaccines are available to prevent some types of pneumonia and the flu.",
      "• Make sure children get vaccinated.",
      "• Practice good hygiene: Wash your hands regularly or use an alcohol-based hand sanitizer.",
      "• Don't smoke: Smoking damages your lungs' natural defenses against respiratory infections.",
      "• Keep your immune system strong: Get enough sleep, exercise regularly and eat a healthy diet."
    ]
    let prevY = yPos + 8
    prevention.forEach(prev => {
      const splitPrev = doc.splitTextToSize(prev, colWidth)
      doc.text(splitPrev, margin + colWidth + margin, prevY)
      const dim = doc.getTextDimensions(splitPrev)
      prevY += dim.h + 3
    })

    // --- DISCLAIMER ---
    doc.setDrawColor(200, 200, 200)
    doc.line(margin, pageHeight - 25, pageWidth - margin, pageHeight - 25)

    doc.setFontSize(8)
    doc.setTextColor(100, 100, 100)
    const disclaimer = "DISCLAIMER: This report is generated by an AI system (PneumoVision v2.0) and is intended for screening assistance only. It does not constitute a final medical diagnosis. All results should be verified by a qualified healthcare professional."
    const splitDisclaimer = doc.splitTextToSize(disclaimer, pageWidth - (margin * 2))
    doc.text(splitDisclaimer, margin, pageHeight - 15)

    doc.save('pneumovision_report.pdf')
  }

  // --- VIEWS ---

  const IdleView = () => (
    <motion.div
      key="idle"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="flex flex-col items-center justify-center min-h-[60vh]"
    >
      <div
        {...getRootProps()}
        onClick={() => fileInputRef.current?.click()}
        className={`relative w-full max-w-2xl aspect-video border-2 border-dashed rounded-3xl flex flex-col items-center justify-center cursor-pointer transition-all duration-500 group overflow-hidden ${isDragActive ? 'border-cyan-500 bg-cyan-500/10' : 'border-slate-700 hover:border-cyan-500/50 hover:bg-slate-800/50'
          }`}
      >
        <input {...getInputProps()} />
        {/* Native File Input Fallback */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          className="hidden"
          accept="image/png, image/jpeg, image/jpg"
        />

        {/* Animated Grid Background */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.05)_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_100%)] pointer-events-none" />

        <div className="relative z-10 flex flex-col items-center">
          <div className="w-24 h-24 bg-slate-800/80 backdrop-blur-sm rounded-full flex items-center justify-center mb-8 border border-slate-600 group-hover:border-cyan-400 group-hover:shadow-[0_0_30px_rgba(34,211,238,0.3)] transition-all duration-300">
            <Upload className="w-10 h-10 text-slate-400 group-hover:text-cyan-400 transition-colors" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-3 font-mono tracking-tight">INITIATE SCAN</h2>
          <p className="text-slate-400 text-lg mb-8">Drop X-Ray Imagery to Begin Analysis</p>
          <div className="flex items-center space-x-4 text-xs font-mono text-cyan-500/70">
            <span className="border border-cyan-500/20 px-3 py-1 rounded-full">DICOM</span>
            <span className="border border-cyan-500/20 px-3 py-1 rounded-full">JPEG</span>
            <span className="border border-cyan-500/20 px-3 py-1 rounded-full">PNG</span>
          </div>

          {error && (
            <div className="mt-6 p-3 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center space-x-2 max-w-md">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-200 text-sm text-left">{error}</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )

  const ReadyView = () => (
    <motion.div
      key="ready"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex flex-col items-center justify-center min-h-[60vh] max-w-4xl mx-auto w-full"
    >
      <div className="w-full bg-slate-900/50 border border-white/10 rounded-3xl overflow-hidden backdrop-blur-sm shadow-2xl">
        <div className="grid md:grid-cols-2 gap-0">
          {/* Image Preview */}
          <div className="relative h-96 bg-black flex items-center justify-center overflow-hidden border-r border-white/5">
            <img src={preview} alt="Preview" className="max-w-full max-h-full object-contain opacity-90" />
            <div className="absolute inset-0 bg-[linear-gradient(transparent_2px,black_3px)] bg-[size:100%_4px] opacity-20 pointer-events-none" />
            <button
              onClick={handleReset}
              className="absolute top-4 left-4 bg-black/50 hover:bg-red-500/80 text-white p-2 rounded-full backdrop-blur-md transition-colors border border-white/10"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Controls */}
          <div className="p-10 flex flex-col justify-center items-start space-y-8">
            <div>
              <div className="flex items-center space-x-2 text-cyan-400 mb-2">
                <FileSearch className="w-5 h-5" />
                <span className="font-mono text-sm tracking-wider">FILE DETECTED</span>
              </div>
              <h3 className="text-2xl font-bold text-white font-mono break-all">{file.name}</h3>
              <p className="text-slate-400 text-sm mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>

            <div className="w-full space-y-4">
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 border-b border-white/10 pb-2">
                <span>SYSTEM STATUS</span>
                <span className="text-green-400">ONLINE</span>
              </div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-500 border-b border-white/10 pb-2">
                <span>MODEL</span>
                <span className="text-cyan-400">ResNet50 v2.1</span>
              </div>
            </div>

            {error && (
              <div className="w-full bg-red-500/10 border border-red-500/30 rounded-lg p-4 flex items-center space-x-3">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            <button
              onClick={handlePredict}
              className="w-full group relative overflow-hidden bg-cyan-500 hover:bg-cyan-400 text-black font-bold py-4 rounded-xl transition-all transform hover:scale-[1.02] active:scale-[0.98]"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
              <span className="relative flex items-center justify-center space-x-2">
                <Cpu className="w-5 h-5" />
                <span>RUN DIAGNOSTICS</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  )

  const ScanningView = () => (
    <motion.div
      key="scanning"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-40 bg-black/90 backdrop-blur-xl flex flex-col items-center justify-center"
    >
      <div className="w-full max-w-md space-y-8 p-6">
        <div className="relative w-32 h-32 mx-auto">
          <div className="absolute inset-0 border-4 border-slate-800 rounded-full" />
          <div
            className="absolute inset-0 border-4 border-cyan-500 rounded-full border-t-transparent animate-spin"
          />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-2xl font-bold text-cyan-400 font-mono">{scanProgress}%</span>
          </div>
        </div>

        <div className="space-y-2">
          <h3 className="text-2xl font-bold text-white text-center font-mono animate-pulse">ANALYZING PATTERNS...</h3>
          <div className="flex justify-between text-xs font-mono text-cyan-500/70">
            <span>LAYER_EXTRACTION</span>
            <span>[PROCESSING]</span>
          </div>
          <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-cyan-500"
              initial={{ width: 0 }}
              animate={{ width: `${scanProgress}%` }}
            />
          </div>
          <div className="font-mono text-xs text-slate-500 text-center pt-4">
            PLEASE WAIT WHILE NEURAL NETWORK PROCESSES IMAGE DATA
          </div>
        </div>
      </div>
    </motion.div>
  )

  const ResultView = () => (
    <motion.div
      key="result"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="w-full max-w-7xl mx-auto"
    >
      {/* Result Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <div className={`w-3 h-3 rounded-full animate-pulse ${result.prediction.is_pneumonia ? 'bg-red-500' : 'bg-green-500'}`} />
            <h2 className="text-3xl font-bold text-white font-mono">DIAGNOSTIC REPORT</h2>
          </div>
          <p className="text-slate-400 font-mono text-sm">ID: {Math.random().toString(36).substr(2, 9).toUpperCase()} • {new Date().toLocaleDateString()}</p>
        </div>
        <button
          onClick={handleReset}
          className="flex items-center space-x-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded-full text-white transition-all group"
        >
          <RefreshCw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-500" />
          <span className="font-mono text-sm">NEW SCAN</span>
        </button>
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-12 gap-8">

        {/* Left Column: Visuals (8 cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Main Hero Image */}
          <div
            className="relative aspect-[4/3] bg-black rounded-3xl overflow-hidden border border-white/10 group cursor-zoom-in"
            onClick={() => setSelectedImage({ src: result.gradcam.overlay, title: "DIAGNOSTIC OVERLAY" })}
          >
            <img
              src={result.gradcam.overlay}
              alt="Diagnostic Overlay"
              className="w-full h-full object-contain"
            />
            <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full border border-white/10">
              <span className="text-xs font-mono text-cyan-400">PRIMARY_VIEW :: OVERLAY</span>
            </div>
            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/20 backdrop-blur-[2px]">
              <Maximize2 className="w-12 h-12 text-white drop-shadow-lg" />
            </div>
          </div>

          {/* Thumbnails */}
          <div className="grid grid-cols-2 gap-4">
            {[
              { title: "ORIGINAL INPUT", img: result.gradcam.original },
              { title: "HEATMAP LAYER", img: result.gradcam.heatmap }
            ].map((item, idx) => (
              <div
                key={idx}
                className="relative aspect-video bg-black rounded-xl overflow-hidden border border-white/10 group cursor-zoom-in"
                onClick={() => setSelectedImage({ src: item.img, title: item.title })}
              >
                <img src={item.img} alt={item.title} className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-3">
                  <span className="text-xs font-mono text-slate-300">{item.title}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Data & Notes (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Prediction Card */}
          <div className={`p-6 rounded-2xl border-2 ${result.prediction.is_pneumonia
            ? 'bg-red-500/10 border-red-500/50'
            : 'bg-green-500/10 border-green-500/50'
            }`}>
            <h3 className="text-sm font-mono text-slate-400 mb-1">DETECTED CONDITION</h3>
            <div className="text-3xl font-bold text-white mb-4 tracking-tight">
              {result.prediction.is_pneumonia ? 'PNEUMONIA' : 'NORMAL'}
            </div>
            <div className="w-full bg-slate-900/50 rounded-full h-2 mb-2 overflow-hidden">
              <div
                className={`h-full ${result.prediction.is_pneumonia ? 'bg-red-500' : 'bg-green-500'}`}
                style={{ width: `${result.prediction.confidence * 100}%` }}
              />
            </div>
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">CONFIDENCE SCORE</span>
              <span className="text-white">{(result.prediction.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Clinical Notes */}
          <div className="glass-dark rounded-2xl p-6 border border-white/10 space-y-6">
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <Stethoscope className="w-4 h-4 text-blue-400" />
                <h4 className="text-xs font-bold text-blue-400 font-mono">PHYSICIAN NOTE</h4>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                {result.prediction.is_pneumonia
                  ? "Opacity detected. Examine highlighted regions in RED on the heatmap for consolidation. Correlate with clinical history."
                  : "No significant opacities detected. Review for other conditions if symptomatic."}
              </p>
            </div>

            <div className="h-px bg-white/10" />

            <div>
              <div className="flex items-center space-x-2 mb-3">
                <User className={`w-4 h-4 ${result.prediction.is_pneumonia ? 'text-red-400' : 'text-green-400'}`} />
                <h4 className={`text-xs font-bold font-mono ${result.prediction.is_pneumonia ? 'text-red-400' : 'text-green-400'}`}>PATIENT ADVISORY</h4>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                {result.prediction.is_pneumonia
                  ? "Potential signs of pneumonia. Consult a pulmonologist immediately for confirmation."
                  : "No signs of pneumonia. Continue to monitor health. Consult a doctor if symptoms persist."}
              </p>
            </div>
          </div>

          {/* Export Action */}
          <button
            onClick={generateReport}
            className="w-full py-4 bg-slate-800 hover:bg-slate-700 border border-slate-600 text-white rounded-xl transition-all flex items-center justify-center group"
          >
            <Download className="w-4 h-4 mr-2 text-cyan-500 group-hover:scale-110 transition-transform" />
            <span className="font-mono text-sm">DOWNLOAD FULL REPORT</span>
          </button>
        </div>
      </div>
    </motion.div>
  )

  return (
    <div className="max-w-7xl mx-auto pt-12 pb-20 px-6 min-h-screen flex flex-col">
      {/* Portal Modal */}
      {createPortal(
        <AnimatePresence>
          {selectedImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/95 backdrop-blur-md p-4"
              onClick={() => setSelectedImage(null)}
            >
              <button
                onClick={() => setSelectedImage(null)}
                className="fixed top-6 right-6 bg-white/10 hover:bg-red-500 text-white p-3 rounded-full transition-colors z-[10000] backdrop-blur-sm border border-white/10"
              >
                <X className="w-6 h-6" />
              </button>
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="relative max-w-7xl max-h-[90vh] flex flex-col items-center justify-center"
                onClick={(e) => e.stopPropagation()}
              >
                <img
                  src={selectedImage.src}
                  alt={selectedImage.title}
                  className="max-w-full max-h-full object-contain rounded-lg shadow-2xl border border-white/10"
                />
                <div className="mt-4 text-center">
                  <h3 className="text-xl font-bold text-white font-mono">{selectedImage.title}</h3>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}

      {/* Main Header (Only show in Idle/Ready states) */}
      {viewState !== 'result' && viewState !== 'scanning' && (
        <div className="text-center mb-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center space-x-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full px-4 py-1.5 mb-6 backdrop-blur-sm"
          >
            <Scan className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400 text-sm font-mono tracking-wider">DIAGNOSTIC TERMINAL V2.0</span>
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-5xl font-bold text-white mb-4 font-mono"
          >
            NEURAL SCANNER
          </motion.h1>
        </div>
      )}

      {/* View Switcher */}
      <AnimatePresence mode="wait">
        {viewState === 'idle' && <IdleView key="idle" />}
        {viewState === 'ready' && <ReadyView key="ready" />}
        {viewState === 'scanning' && <ScanningView key="scanning" />}
        {viewState === 'result' && <ResultView key="result" />}
      </AnimatePresence>
    </div>
  )
}

export default UploadPage
