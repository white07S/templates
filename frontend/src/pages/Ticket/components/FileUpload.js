import React, { useState, useRef } from 'react';
import { Upload, X, FileText, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { validateFile } from '../../../utils/fileValidation';
import { formatFileSize } from '../../../utils/helpers';
import './FileUpload.css';

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
    <div className="file-upload">
      <label className="selector-label">
        Upload File
        <span className="input-required">*</span>
      </label>

      {!file ? (
        <div
          className={`upload-area ${dragActive ? 'upload-area-active' : ''} ${disabled ? 'upload-area-disabled' : ''}`}
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
            className="file-input-hidden"
          />
          
          <Upload size={48} className="upload-icon" />
          <p className="upload-text">
            Drag and drop your file here, or click to browse
          </p>
          <p className="upload-hint">
            Supported formats: XLSX, CSV (max 50MB)
          </p>
        </div>
      ) : (
        <motion.div
          className="file-preview"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <div className="file-info">
            <FileText size={24} className="file-icon" />
            <div className="file-details">
              <p className="file-name">{file.name}</p>
              <p className="file-size">{formatFileSize(file.size)}</p>
            </div>
          </div>
          <button
            type="button"
            className="file-remove"
            onClick={handleRemove}
            disabled={disabled}
          >
            <X size={20} />
          </button>
        </motion.div>
      )}

      <AnimatePresence>
        {validationError && (
          <motion.div
            className="upload-error"
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