import React, { useState, useRef } from 'react';
import { Upload, X, FileText, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { validateFile } from '../../../utils/fileValidation';
import { formatFileSize } from '../../../utils/helpers';

const FileUpload = ({ 
  onFileSelect, 
  requiredColumns = [], 
  disabled = false,
  file,
  onRemove 
}) => {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState('');

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (disabled) return;
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = async (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      await handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    setValidationError('');
    
    const validation = await validateFile(file, requiredColumns);
    
    if (!validation.isValid) {
      setValidationError(validation.errors.join(', '));
      return;
    }
    
    onFileSelect(file);
  };

  const handleRemove = () => {
    setValidationError('');
    if (onRemove) onRemove();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-black">
        Upload File
        <span className="text-red-600 ml-1">*</span>
      </label>

      {!file ? (
        <div
          className={`border-2 ${dragActive ? 'border-red-600 bg-red-50' : 'border-gray-300'} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:border-red-600'} p-8 text-center transition-colors`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".xlsx,.csv"
            onChange={handleChange}
            disabled={disabled}
            className="hidden"
          />
          
          <Upload size={48} className="mx-auto mb-4 text-gray-400" />
          <p className="text-black font-medium mb-2">
            Drag and drop your file here, or click to browse
          </p>
          <p className="text-gray-600 text-sm">
            Supported formats: XLSX, CSV (max 50MB)
          </p>
        </div>
      ) : (
        <motion.div
          className="border-2 border-gray-300 p-4 flex items-center justify-between bg-gray-50"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="flex items-center space-x-3">
            <FileText size={24} className="text-red-600" />
            <div>
              <p className="text-black font-medium">{file.name}</p>
              <p className="text-gray-600 text-sm">{formatFileSize(file.size)}</p>
            </div>
          </div>
          <button
            type="button"
            className="p-1 hover:bg-gray-200 transition-colors"
            onClick={handleRemove}
            disabled={disabled}
          >
            <X size={20} className="text-gray-600" />
          </button>
        </motion.div>
      )}

      <AnimatePresence>
        {validationError && (
          <motion.div
            className="flex items-center space-x-2 text-red-600 text-sm"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <AlertCircle size={16} />
            <span>{validationError}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FileUpload;