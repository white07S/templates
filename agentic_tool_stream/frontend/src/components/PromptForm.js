import React, { useState, useEffect } from 'react';
import { X, Plus, Trash2 } from 'lucide-react';

const PromptForm = ({ prompt, onSubmit, onCancel }) => {
  const [formData, setFormData] = useState({
    persona: '',
    task: '',
    if_task_need_data: false,
    data: '',
    response: '',
    keywords_used_for_search: []
  });
  const [newKeyword, setNewKeyword] = useState('');
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (prompt) {
      setFormData({
        persona: prompt.persona || '',
        task: prompt.task || '',
        if_task_need_data: prompt.if_task_need_data || false,
        data: prompt.data || '',
        response: prompt.response || '',
        keywords_used_for_search: prompt.keywords_used_for_search || []
      });
    }
  }, [prompt]);

  const validateForm = () => {
    const newErrors = {};

    if (!formData.persona.trim()) {
      newErrors.persona = 'Persona is required';
    }

    if (!formData.task.trim()) {
      newErrors.task = 'Task is required';
    }

    if (!formData.response.trim()) {
      newErrors.response = 'Response format is required';
    }

    if (formData.if_task_need_data && !formData.data?.trim()) {
      newErrors.data = 'Data is required when task needs data';
    }

    if (formData.keywords_used_for_search.length === 0) {
      newErrors.keywords = 'At least one keyword is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const handleAddKeyword = () => {
    if (newKeyword.trim() && !formData.keywords_used_for_search.includes(newKeyword.trim())) {
      setFormData({
        ...formData,
        keywords_used_for_search: [...formData.keywords_used_for_search, newKeyword.trim()]
      });
      setNewKeyword('');
      setErrors({ ...errors, keywords: undefined });
    }
  };

  const handleRemoveKeyword = (index) => {
    setFormData({
      ...formData,
      keywords_used_for_search: formData.keywords_used_for_search.filter((_, i) => i !== index)
    });
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddKeyword();
    }
  };

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">
          {prompt ? 'Edit Prompt' : 'Create New Prompt'}
        </h2>
        <button
          onClick={onCancel}
          className="p-2 hover:bg-gray-100 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-semibold mb-2">
            Persona <span className="text-red-600">*</span>
          </label>
          <input
            type="text"
            value={formData.persona}
            onChange={(e) => setFormData({ ...formData, persona: e.target.value })}
            placeholder="e.g., Expert Python Developer, Marketing Specialist..."
            className={`w-full p-3 border-2 ${
              errors.persona ? 'border-red-600' : 'border-black'
            } focus:outline-none focus:ring-2 focus:ring-gray-400`}
          />
          {errors.persona && (
            <p className="text-red-600 text-sm mt-1">{errors.persona}</p>
          )}
        </div>

        <div>
          <label className="block text-sm font-semibold mb-2">
            Task <span className="text-red-600">*</span>
          </label>
          <textarea
            value={formData.task}
            onChange={(e) => setFormData({ ...formData, task: e.target.value })}
            placeholder="Describe the task this prompt should accomplish..."
            rows={3}
            className={`w-full p-3 border-2 ${
              errors.task ? 'border-red-600' : 'border-black'
            } focus:outline-none focus:ring-2 focus:ring-gray-400`}
          />
          {errors.task && (
            <p className="text-red-600 text-sm mt-1">{errors.task}</p>
          )}
        </div>

        <div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={formData.if_task_need_data}
              onChange={(e) => setFormData({ ...formData, if_task_need_data: e.target.checked })}
              className="w-4 h-4 border-2 border-black"
            />
            <span className="text-sm font-semibold">Task requires additional data input</span>
          </label>
        </div>

        {formData.if_task_need_data && (
          <div>
            <label className="block text-sm font-semibold mb-2">
              Data Format/Example <span className="text-red-600">*</span>
            </label>
            <textarea
              value={formData.data}
              onChange={(e) => setFormData({ ...formData, data: e.target.value })}
              placeholder="Describe or provide example of the data format needed..."
              rows={3}
              className={`w-full p-3 border-2 ${
                errors.data ? 'border-red-600' : 'border-black'
              } focus:outline-none focus:ring-2 focus:ring-gray-400`}
            />
            {errors.data && (
              <p className="text-red-600 text-sm mt-1">{errors.data}</p>
            )}
          </div>
        )}

        <div>
          <label className="block text-sm font-semibold mb-2">
            Expected Response Format <span className="text-red-600">*</span>
          </label>
          <textarea
            value={formData.response}
            onChange={(e) => setFormData({ ...formData, response: e.target.value })}
            placeholder="Describe the expected format or structure of the response..."
            rows={3}
            className={`w-full p-3 border-2 ${
              errors.response ? 'border-red-600' : 'border-black'
            } focus:outline-none focus:ring-2 focus:ring-gray-400`}
          />
          {errors.response && (
            <p className="text-red-600 text-sm mt-1">{errors.response}</p>
          )}
        </div>

        <div>
          <label className="block text-sm font-semibold mb-2">
            Keywords for Search <span className="text-red-600">*</span>
          </label>
          <div className="flex gap-2 mb-2">
            <input
              type="text"
              value={newKeyword}
              onChange={(e) => setNewKeyword(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Add a keyword..."
              className="flex-1 p-2 border-2 border-black focus:outline-none focus:ring-2 focus:ring-gray-400"
            />
            <button
              type="button"
              onClick={handleAddKeyword}
              className="px-4 py-2 bg-black text-white hover:bg-gray-800 transition-colors"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          {errors.keywords && (
            <p className="text-red-600 text-sm mb-2">{errors.keywords}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {formData.keywords_used_for_search.map((keyword, index) => (
              <div
                key={index}
                className="flex items-center gap-1 px-3 py-1 border-2 border-black bg-gray-50"
              >
                <span className="text-sm">{keyword}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveKeyword(index)}
                  className="p-1 hover:bg-gray-200 transition-colors rounded"
                >
                  <Trash2 className="w-3 h-3 text-red-600" />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="flex gap-4 pt-4">
          <button
            type="submit"
            className="flex-1 px-6 py-3 bg-black text-white hover:bg-gray-800 transition-colors font-semibold"
          >
            {prompt ? 'Update Prompt' : 'Create Prompt'}
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="px-6 py-3 border-2 border-black hover:bg-gray-100 transition-colors font-semibold"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
};

export default PromptForm;