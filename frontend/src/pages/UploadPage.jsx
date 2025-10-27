import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { Upload, X, Loader, CheckCircle, AlertCircle, Download } from 'lucide-react'

function UploadPage() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1,
    multiple: false
  })

  const handleRemoveFile = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/predict-with-gradcam', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction')
    } finally {
      setLoading(false)
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

  return (
    <div className="max-w-6xl mx-auto fade-in">
      <h1 className="text-4xl font-bold text-gray-800 mb-2 text-center">
        Upload Chest X-Ray
      </h1>
      <p className="text-gray-600 mb-8 text-center">
        Upload an X-ray image to get AI-powered pneumonia detection with explainable visualization
      </p>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Upload Section */}
        <div>
          <div className="card">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Select Image</h2>

            {!file ? (
              <div
                {...getRootProps()}
                className={`border-3 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${
                  isDragActive
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                {isDragActive ? (
                  <p className="text-blue-600 font-medium">Drop the image here...</p>
                ) : (
                  <>
                    <p className="text-gray-600 font-medium mb-2">
                      Drag & drop an X-ray image here
                    </p>
                    <p className="text-gray-500 text-sm mb-4">or click to browse</p>
                    <button className="btn-secondary text-sm">
                      Browse Files
                    </button>
                    <p className="text-xs text-gray-400 mt-4">
                      Supported formats: PNG, JPG, JPEG (Max 16MB)
                    </p>
                  </>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="relative">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full h-64 object-contain bg-gray-100 rounded-lg"
                  />
                  <button
                    onClick={handleRemoveFile}
                    className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">File:</span> {file.name}
                  </p>
                  <p className="text-sm text-gray-600">
                    <span className="font-medium">Size:</span>{' '}
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="btn-primary w-full"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Analyze Image
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 bg-red-50 border-2 border-red-200 rounded-lg p-4 flex items-start space-x-3">
              <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-800">Error</h3>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div>
          {result ? (
            <div className="space-y-4">
              {/* Prediction Result */}
              <div className={`card ${
                result.prediction.is_pneumonia
                  ? 'bg-gradient-to-br from-red-50 to-orange-50 border-2 border-red-200'
                  : 'bg-gradient-to-br from-green-50 to-teal-50 border-2 border-green-200'
              }`}>
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Prediction Result</h2>
                <div className="text-center">
                  <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
                    result.prediction.is_pneumonia
                      ? 'bg-red-500'
                      : 'bg-green-500'
                  }`}>
                    {result.prediction.is_pneumonia ? (
                      <AlertCircle className="w-10 h-10 text-white" />
                    ) : (
                      <CheckCircle className="w-10 h-10 text-white" />
                    )}
                  </div>
                  <h3 className={`text-3xl font-bold mb-2 ${
                    result.prediction.is_pneumonia ? 'text-red-700' : 'text-green-700'
                  }`}>
                    {result.prediction.class}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    Confidence: {(result.prediction.confidence * 100).toFixed(2)}%
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all duration-500 ${
                        result.prediction.is_pneumonia
                          ? 'bg-gradient-to-r from-red-500 to-orange-500'
                          : 'bg-gradient-to-r from-green-500 to-teal-500'
                      }`}
                      style={{ width: `${result.prediction.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Grad-CAM Visualization */}
              <div className="card">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">
                  Grad-CAM Visualization
                </h2>
                <p className="text-gray-600 text-sm mb-4">
                  Heatmap showing which lung regions influenced the AI's decision
                </p>
                
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold text-gray-700 mb-2">Original X-Ray</h3>
                    <img
                      src={result.gradcam.original}
                      alt="Original"
                      className="w-full rounded-lg border-2 border-gray-200"
                    />
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-2">Heatmap</h3>
                    <img
                      src={result.gradcam.heatmap}
                      alt="Heatmap"
                      className="w-full rounded-lg border-2 border-gray-200"
                    />
                  </div>

                  <div>
                    <h3 className="font-semibold text-gray-700 mb-2">Overlay</h3>
                    <img
                      src={result.gradcam.overlay}
                      alt="Overlay"
                      className="w-full rounded-lg border-2 border-gray-200"
                    />
                  </div>

                  <button
                    onClick={() => downloadImage(result.gradcam.overlay, 'gradcam_result.png')}
                    className="btn-secondary w-full"
                  >
                    <Download className="w-5 h-5 mr-2" />
                    Download Grad-CAM Result
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="card h-full flex items-center justify-center text-center p-12">
              <div>
                <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-12 h-12 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-gray-600 mb-2">
                  No Results Yet
                </h3>
                <p className="text-gray-500">
                  Upload an X-ray image and click "Analyze" to see results
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UploadPage
