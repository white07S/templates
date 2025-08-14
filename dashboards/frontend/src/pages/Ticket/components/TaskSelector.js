import React from 'react';
import { Check } from 'lucide-react';
import { motion } from 'framer-motion';

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
      <div className="p-4 bg-gray-50 border-2 border-gray-200 text-center">
        <p className="text-gray-600">Please select a data type first to see available tasks</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-black">
          Select Tasks
          <span className="text-red-600 ml-1">*</span>
        </label>
        <button
          type="button"
          className="px-3 py-1 text-sm font-medium text-red-600 hover:text-red-700 transition-colors"
          onClick={handleSelectAll}
          disabled={disabled}
        >
          {selectedTasks.length === availableTasks.length ? 'Deselect All' : 'Select All'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {availableTasks.map((task, index) => (
          <motion.label
            key={task}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: index * 0.05 }}
            className={`flex items-center p-3 border-2 border-gray-200 hover:border-red-600 cursor-pointer transition-colors ${
              selectedTasks.includes(task) ? 'bg-red-50 border-red-600' : 'bg-white'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <input
              type="checkbox"
              checked={selectedTasks.includes(task)}
              onChange={() => handleTaskToggle(task)}
              disabled={disabled}
              className="sr-only"
            />
            <div className={`flex items-center justify-center w-5 h-5 border-2 mr-3 ${
              selectedTasks.includes(task) ? 'bg-red-600 border-red-600' : 'border-gray-300'
            }`}>
              {selectedTasks.includes(task) && <Check size={16} className="text-white" />}
            </div>
            <span className="text-black font-medium">{task}</span>
          </motion.label>
        ))}
      </div>

      {selectedTasks.length > 0 && (
        <p className="text-sm text-gray-600">
          {selectedTasks.length} task{selectedTasks.length > 1 ? 's' : ''} selected
        </p>
      )}
    </div>
  );
};

export default TaskSelector;