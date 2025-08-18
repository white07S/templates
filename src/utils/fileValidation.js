import { FILE_FORMATS, MAX_FILE_SIZE } from './constants';

export const validateFileFormat = (file) => {
  const extension = '.' + file.name.split('.').pop().toLowerCase();
  return Object.values(FILE_FORMATS).includes(extension);
};

export const validateFileSize = (file) => {
  return file.size <= MAX_FILE_SIZE;
};

export const readFileHeaders = async (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    if (file.name.endsWith('.csv')) {
      reader.onload = (e) => {
        const text = e.target.result;
        const firstLine = text.split('\n')[0];
        const headers = firstLine.split(',').map(h => h.trim().replace(/"/g, ''));
        resolve(headers);
      };
      reader.readAsText(file);
    } else if (file.name.endsWith('.xlsx')) {
      // For XLSX files, we'll validate on the server side
      resolve([]);
    } else {
      reject(new Error('Unsupported file format'));
    }
  });
};

export const validateFile = async (file, requiredColumns = []) => {
  const errors = [];

  if (!validateFileFormat(file)) {
    errors.push('File must be in XLSX or CSV format');
  }

  if (!validateFileSize(file)) {
    errors.push(`File size must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB`);
  }

  if (file.name.endsWith('.csv') && requiredColumns.length > 0) {
    try {
      const headers = await readFileHeaders(file);
      const missingColumns = requiredColumns.filter(col => !headers.includes(col));
      if (missingColumns.length > 0) {
        errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
      }
    } catch (error) {
      errors.push('Unable to read file headers');
    }
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};