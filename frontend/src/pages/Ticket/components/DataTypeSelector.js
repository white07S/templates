import React from 'react';
import { ChevronDown } from 'lucide-react';
import { formatDataType } from '../../../utils/helpers';
import './DataTypeSelector.css';

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
    <div className="data-type-selector">
      <label className="selector-label">
        Data Type
        <span className="input-required">*</span>
      </label>
      
      <div className="selector-wrapper">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className="selector-input"
        >
          <option value="">Select a data type</option>
          {dataTypes.map((dataType) => (
            <option key={dataType.name} value={dataType.name}>
              {formatDataType(dataType.name)}
            </option>
          ))}
        </select>
        <ChevronDown className="selector-icon" size={20} />
      </div>

      {selectedDataType && (
        <div className="selector-info">
          <p className="info-title">Required columns:</p>
          <ul className="info-list">
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