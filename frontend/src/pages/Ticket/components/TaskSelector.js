import React from 'react';
import { Check } from 'lucide-react';
import './TaskSelector.css';

const TaskSelector = ({ 
  availableTasks = [], 
  selectedTasks = [], 
  onChange,
  disabled = false 
}) => {
  const handleTaskToggle = (task) => {
    if (selectedTasks.includes(task)) {
      onChange(selectedTasks.filter(t => t !== task));
    } else {
      onChange([...selectedTasks, task]);
    }
  };

  const handleSelectAll = () => {
    if (selectedTasks.length === availableTasks.length) {
      onChange([]);
    } else {
      onChange([...availableTasks]);
    }
  };

  if (availableTasks.length === 0) {
    return (
      <div className="task-selector-empty">
        <p className="text-small">Please select a data type first to see available tasks</p>
      </div>
    );
  }

  return (
    <div className="task-selector">
      <div className="task-selector-header">
        <label className="selector-label">
          Select Tasks
          <span className="input-required">*</span>
        </label>
        <button
          type="button"
          className="select-all-btn"
          onClick={handleSelectAll}
          disabled={disabled}
        >
          {selectedTasks.length === availableTasks.length ? 'Deselect All' : 'Select All'}
        </button>
      </div>

      <div className="task-list">
        {availableTasks.map((task) => (
          <label
            key={task}
            className={`task-item ${disabled ? 'task-item-disabled' : ''}`}
          >
            <input
              type="checkbox"
              checked={selectedTasks.includes(task)}
              onChange={() => handleTaskToggle(task)}
              disabled={disabled}
              className="task-checkbox"
            />
            <span className="task-checkbox-custom">
              {selectedTasks.includes(task) && <Check size={16} />}
            </span>
            <span className="task-name">{task}</span>
          </label>
        ))}
      </div>

      {selectedTasks.length > 0 && (
        <p className="task-count">
          {selectedTasks.length} task{selectedTasks.length > 1 ? 's' : ''} selected
        </p>
      )}
    </div>
  );
};

export default TaskSelector;