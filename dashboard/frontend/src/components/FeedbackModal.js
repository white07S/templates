import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send, MessageSquare, Star, ThumbsUp, ThumbsDown, AlertCircle } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import { apiService } from '../services/api';

const FeedbackTypeSelector = ({ selectedType, onTypeChange, datasetType }) => {
  // Different feedback types for each dataset
  const getFeedbackTypes = () => {
    switch(datasetType) {
      case 'external_loss':
        return [
          { value: 'severity', label: 'Severity Level', icon: AlertCircle, description: 'Rate loss severity (1-5)' },
          { value: 'verification', label: 'Verify Data', icon: ThumbsUp, description: 'Confirm data accuracy' },
          { value: 'impact', label: 'Impact Assessment', icon: MessageSquare, description: 'Describe business impact' },
          { value: 'correction', label: 'Data Correction', icon: Star, description: 'Suggest corrections' }
        ];
      case 'internal_loss':
        return [
          { value: 'priority', label: 'Priority', icon: Star, description: 'Set priority level (1-5)' },
          { value: 'category', label: 'Categorization', icon: ThumbsUp, description: 'Confirm category' },
          { value: 'root_cause', label: 'Root Cause', icon: MessageSquare, description: 'Identify root cause' },
          { value: 'prevention', label: 'Prevention', icon: AlertCircle, description: 'Suggest prevention measures' }
        ];
      case 'issues':
        return [
          { value: 'status', label: 'Status Update', icon: ThumbsUp, description: 'Update issue status' },
          { value: 'urgency', label: 'Urgency', icon: AlertCircle, description: 'Set urgency level (1-5)' },
          { value: 'resolution', label: 'Resolution', icon: MessageSquare, description: 'Provide resolution details' },
          { value: 'escalation', label: 'Escalation', icon: Star, description: 'Escalation needed?' }
        ];
      case 'controls':
        return [
          { value: 'effectiveness', label: 'Effectiveness', icon: Star, description: 'Rate effectiveness (1-5)' },
          { value: 'compliance', label: 'Compliance', icon: ThumbsUp, description: 'Compliance status' },
          { value: 'improvement', label: 'Improvement', icon: MessageSquare, description: 'Suggest improvements' },
          { value: 'risk_assessment', label: 'Risk Level', icon: AlertCircle, description: 'Assess risk level' }
        ];
      default:
        return [
          { value: 'rating', label: 'Rating', icon: Star, description: 'Rate from 1 to 5 stars' },
          { value: 'approval', label: 'Approval', icon: ThumbsUp, description: 'Approve or reject' },
          { value: 'text', label: 'Comment', icon: MessageSquare, description: 'Free text feedback' },
          { value: 'issue', label: 'Report Issue', icon: AlertCircle, description: 'Report a problem' }
        ];
    }
  };

  const feedbackTypes = getFeedbackTypes();

  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-3">Feedback Type</h3>
      <div className="grid grid-cols-2 gap-3">
        {feedbackTypes.map(type => {
          const Icon = type.icon;
          return (
            <button
              key={type.value}
              onClick={() => onTypeChange(type.value)}
              className={`
                p-4 border-2 transition-all duration-200 text-left
                ${selectedType === type.value
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
                }
              `}
            >
              <div className="flex items-center mb-2">
                <Icon className={`h-5 w-5 mr-2 ${selectedType === type.value ? 'text-primary-600' : 'text-gray-400'}`} />
                <span className={`font-medium ${selectedType === type.value ? 'text-primary-900' : 'text-gray-700'}`}>
                  {type.label}
                </span>
              </div>
              <p className={`text-sm ${selectedType === type.value ? 'text-primary-700' : 'text-gray-500'}`}>
                {type.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
};

const RatingInput = ({ value, onChange }) => {
  return (
    <div className="flex items-center space-x-2">
      {[1, 2, 3, 4, 5].map(star => (
        <button
          key={star}
          onClick={() => onChange(star.toString())}
          className={`p-1 transition-colors ${
            parseInt(value) >= star ? 'text-yellow-500' : 'text-gray-300 hover:text-yellow-300'
          }`}
        >
          <Star className="h-8 w-8 fill-current" />
        </button>
      ))}
      <span className="ml-3 text-gray-600">
        {value ? `${value}/5 stars` : 'Select rating'}
      </span>
    </div>
  );
};

const ApprovalInput = ({ value, onChange }) => {
  return (
    <div className="flex items-center space-x-4">
      <button
        onClick={() => onChange('approved')}
        className={`
          flex items-center px-4 py-3 border-2 transition-all duration-200
          ${value === 'approved'
            ? 'border-green-500 bg-green-50 text-green-700'
            : 'border-gray-200 hover:border-green-300 text-gray-600'
          }
        `}
      >
        <ThumbsUp className="h-5 w-5 mr-2" />
        Approve
      </button>
      
      <button
        onClick={() => onChange('rejected')}
        className={`
          flex items-center px-4 py-3 border-2 transition-all duration-200
          ${value === 'rejected'
            ? 'border-red-500 bg-red-50 text-red-700'
            : 'border-gray-200 hover:border-red-300 text-gray-600'
          }
        `}
      >
        <ThumbsDown className="h-5 w-5 mr-2" />
        Reject
      </button>
    </div>
  );
};

const FeedbackModal = () => {
  const { state, actions } = useDashboard();
  const [feedbackType, setFeedbackType] = useState('text');
  const [feedbackValue, setFeedbackValue] = useState('');
  const [additionalNotes, setAdditionalNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [existingFeedback, setExistingFeedback] = useState([]);
  
  const isOpen = !!state.feedbackRecord;
  const record = state.feedbackRecord;

  // Load existing feedback when modal opens
  useEffect(() => {
    if (isOpen && record) {
      loadExistingFeedback();
    }
  }, [isOpen, record?.id]);

  const loadExistingFeedback = async () => {
    try {
      const feedback = await apiService.getFeedback(record.id);
      setExistingFeedback(feedback);
    } catch (error) {
      console.error('Failed to load existing feedback:', error);
    }
  };

  const closeModal = () => {
    actions.setFeedbackRecord(null);
    setFeedbackType('text');
    setFeedbackValue('');
    setAdditionalNotes('');
    setExistingFeedback([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!feedbackValue.trim()) return;

    setIsSubmitting(true);
    try {
      const feedbackData = {
        feedback_type: feedbackType,
        value: feedbackValue,
        additional_notes: additionalNotes.trim() || null
      };

      await apiService.submitFeedback(record.id, feedbackData);
      
      // Reload existing feedback
      await loadExistingFeedback();
      
      // Reset form
      setFeedbackValue('');
      setAdditionalNotes('');
      
      // Show success message or close modal
      // closeModal();
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // Handle error (show toast, etc.)
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderFeedbackInput = () => {
    // Custom inputs based on dataset and feedback type
    const datasetType = record?.dataset_type;
    
    // Rating types
    if (['rating', 'severity', 'priority', 'urgency', 'effectiveness'].includes(feedbackType)) {
      return <RatingInput value={feedbackValue} onChange={setFeedbackValue} />;
    }
    
    // Approval/verification types
    if (['approval', 'verification', 'category', 'compliance', 'status', 'escalation'].includes(feedbackType)) {
      const getApprovalOptions = () => {
        switch(feedbackType) {
          case 'status':
            return ['open', 'in_progress', 'resolved', 'closed'];
          case 'compliance':
            return ['compliant', 'non_compliant', 'partial', 'review_needed'];
          case 'escalation':
            return ['yes', 'no', 'pending_review'];
          default:
            return ['approved', 'rejected'];
        }
      };
      
      const options = getApprovalOptions();
      return (
        <div className="flex flex-wrap gap-3">
          {options.map(option => (
            <button
              key={option}
              onClick={() => setFeedbackValue(option)}
              className={`
                px-4 py-2 border-2 transition-all duration-200 capitalize
                ${feedbackValue === option
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-200 hover:border-gray-300 text-gray-600'
                }
              `}
            >
              {option.replace('_', ' ')}
            </button>
          ))}
        </div>
      );
    }
    
    // Text inputs
    return (
      <textarea
        value={feedbackValue}
        onChange={(e) => setFeedbackValue(e.target.value)}
        placeholder={getPlaceholder()}
        className="w-full px-3 py-2 border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
        rows={4}
      />
    );
    
    function getPlaceholder() {
      const placeholders = {
        'impact': 'Describe the business impact of this loss...',
        'correction': 'Suggest data corrections...',
        'root_cause': 'Identify the root cause...',
        'prevention': 'Suggest prevention measures...',
        'resolution': 'Provide resolution details...',
        'improvement': 'Suggest improvements...',
        'risk_assessment': 'Assess the risk level and explain...'
      };
      return placeholders[feedbackType] || 'Enter your feedback...';
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
          onClick={closeModal}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-white shadow-2xl w-[90vw] h-[85vh] max-w-7xl overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  Feedback for Record #{record?.id}
                </h2>
                <p className="text-sm text-gray-600 mt-1">
                  {record?.description?.substring(0, 100)}...
                </p>
              </div>
              
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(85vh - 88px)' }}>
              {/* Existing Feedback */}
              {existingFeedback.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Previous Feedback</h3>
                  <div className="space-y-3">
                    {existingFeedback.map((feedback, index) => (
                      <div key={feedback.id} className="p-4 bg-gray-50">
                        <div className="flex items-center justify-between mb-2">
                          <span className="inline-flex items-center px-2 py-1 text-xs font-medium bg-gray-200 text-gray-800 capitalize">
                            {feedback.feedback_type}
                          </span>
                          <span className="text-xs text-gray-500">
                            {new Date(feedback.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-gray-700">{feedback.value}</p>
                        {feedback.additional_notes && (
                          <p className="text-sm text-gray-600 mt-1 italic">
                            Note: {feedback.additional_notes}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* New Feedback Form */}
              <form onSubmit={handleSubmit}>
                <FeedbackTypeSelector
                  selectedType={feedbackType}
                  onTypeChange={setFeedbackType}
                  datasetType={record?.dataset_type}
                />

                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Your Feedback</h3>
                  {renderFeedbackInput()}
                </div>

                {/* Additional Notes */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Additional Notes (Optional)</h3>
                  <textarea
                    value={additionalNotes}
                    onChange={(e) => setAdditionalNotes(e.target.value)}
                    placeholder="Any additional context or notes..."
                    className="w-full px-3 py-2 border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    rows={3}
                  />
                </div>

                {/* Submit Button */}
                <div className="flex justify-end space-x-3">
                  <button
                    type="button"
                    onClick={closeModal}
                    className="btn-secondary"
                    disabled={isSubmitting}
                  >
                    Cancel
                  </button>
                  
                  <button
                    type="submit"
                    disabled={!feedbackValue.trim() || isSubmitting}
                    className="btn-primary flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                        Submitting...
                      </>
                    ) : (
                      <>
                        <Send className="h-4 w-4 mr-2" />
                        Submit Feedback
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default FeedbackModal;