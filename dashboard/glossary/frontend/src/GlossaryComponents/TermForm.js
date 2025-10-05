import React, { useState, useEffect } from 'react';

const TermForm = ({ term, onSubmit, onCancel, userId }) => {
  const [formData, setFormData] = useState({
    term: '',
    definition: '',
    synonyms: []
  });
  const [synonymInput, setSynonymInput] = useState('');
  const [errors, setErrors] = useState({});

  useEffect(() => {
    if (term) {
      setFormData({
        term: term.term || '',
        definition: term.definition || '',
        synonyms: term.synonyms || []
      });
    }
  }, [term]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error for this field when user types
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validate = () => {
    const newErrors = {};

    if (!formData.term.trim()) {
      newErrors.term = 'Term is required';
    }

    if (!formData.definition.trim()) {
      newErrors.definition = 'Definition is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleAddSynonym = () => {
    if (synonymInput.trim()) {
      setFormData(prev => ({
        ...prev,
        synonyms: [...prev.synonyms, synonymInput.trim()]
      }));
      setSynonymInput('');
    }
  };

  const handleRemoveSynonym = (index) => {
    setFormData(prev => ({
      ...prev,
      synonyms: prev.synonyms.filter((_, i) => i !== index)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    try {
      await onSubmit(formData);
      setFormData({ term: '', definition: '', synonyms: [] });
      setSynonymInput('');
      setErrors({});
    } catch (error) {
      setErrors({ submit: error.message });
    }
  };

  return (
    <div className="bg-white border border-gray-300 p-6">
      <h2 className="text-xl font-bold mb-4 text-gray-800">
        {term ? 'Edit Term' : 'Create New Term'}
      </h2>

      {errors.submit && (
        <div className="mb-4 p-3 bg-red-50 border border-red-400 text-red-700">
          {errors.submit}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="term" className="block text-sm font-medium mb-2 text-gray-700">
            Term *
          </label>
          <input
            type="text"
            id="term"
            name="term"
            value={formData.term}
            onChange={handleChange}
            className={`w-full px-3 py-2 border ${
              errors.term ? 'border-red-400' : 'border-gray-300'
            } focus:outline-none focus:ring-1 focus:ring-gray-400`}
            placeholder="Enter the term"
          />
          {errors.term && (
            <p className="mt-1 text-sm text-red-600">{errors.term}</p>
          )}
        </div>

        <div className="mb-4">
          <label htmlFor="definition" className="block text-sm font-medium mb-2 text-gray-700">
            Definition *
          </label>
          <textarea
            id="definition"
            name="definition"
            value={formData.definition}
            onChange={handleChange}
            rows={4}
            className={`w-full px-3 py-2 border ${
              errors.definition ? 'border-red-400' : 'border-gray-300'
            } focus:outline-none focus:ring-1 focus:ring-gray-400`}
            placeholder="Enter the definition"
          />
          {errors.definition && (
            <p className="mt-1 text-sm text-red-600">{errors.definition}</p>
          )}
        </div>

        <div className="mb-4">
          <label htmlFor="synonyms" className="block text-sm font-medium mb-2 text-gray-700">
            Synonyms
          </label>
          <div className="flex gap-2 mb-2">
            <input
              type="text"
              id="synonyms"
              value={synonymInput}
              onChange={(e) => setSynonymInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleAddSynonym();
                }
              }}
              className="flex-1 px-3 py-2 border border-gray-300 focus:outline-none focus:ring-1 focus:ring-gray-400"
              placeholder="Enter a synonym and press Enter or click Add"
            />
            <button
              type="button"
              onClick={handleAddSynonym}
              className="px-4 py-2 bg-gray-500 text-white hover:bg-gray-600 transition-colors"
            >
              Add
            </button>
          </div>
          {formData.synonyms.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {formData.synonyms.map((synonym, index) => (
                <span
                  key={index}
                  className="inline-flex items-center gap-1 px-3 py-1 bg-gray-100 border border-gray-300 text-sm"
                >
                  {synonym}
                  <button
                    type="button"
                    onClick={() => handleRemoveSynonym(index)}
                    className="text-red-500 hover:text-red-700 font-bold"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex gap-3">
          <button
            type="submit"
            className="px-6 py-2 bg-gray-600 text-white hover:bg-gray-700 transition-colors"
          >
            {term ? 'Update' : 'Create'} Term
          </button>
          <button
            type="button"
            onClick={onCancel}
            className="px-6 py-2 bg-white border border-gray-400 text-gray-700 hover:bg-gray-100 transition-colors"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
};

export default TermForm;