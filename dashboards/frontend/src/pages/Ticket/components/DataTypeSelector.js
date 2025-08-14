import React from 'react';
import { ChevronDown } from 'lucide-react';
import { formatDataType } from '../../../utils/helpers';

const DataTypeSelector = ({ value, onChange, dataTypes, disabled = false }) => {
  const selectedDataType = dataTypes.find(dt => dt.name === value);

  const getRequiredColumns = (dataTypeName) => {
    const columnMap = {
      controls: ['Control Description', 'Control ID'],
      issues: ['Issue Description', 'Issue ID'],
      external_loss: ['Loss Description', 'Loss ID'],
      internal_loss: ['Loss Description', 'Loss ID'],
      orx_scenarios: ['Scenario Description', 'Scenario ID']
    };
    return columnMap[dataTypeName] || [];
  };

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-black">
        Data Type
        <span className="text-red-600 ml-1">*</span>
      </label>
      
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="w-full px-3 py-2 border-2 border-gray-300 bg-white text-black focus:outline-none focus:border-red-600 transition-colors appearance-none"
        >
          <option value="">Select a data type</option>
          {dataTypes.map((dataType) => (
            <option key={dataType.name} value={dataType.name}>
              {formatDataType(dataType.name)}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 pointer-events-none" size={20} />
      </div>

      {selectedDataType && (
        <div className="mt-3 p-3 bg-gray-50 border-l-4 border-red-600">
          <p className="font-medium text-black mb-1">Required columns:</p>
          <ul className="list-disc list-inside text-gray-600 text-sm">
            {getRequiredColumns(selectedDataType.name).map((col) => (
              <li key={col}>{col}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DataTypeSelector;