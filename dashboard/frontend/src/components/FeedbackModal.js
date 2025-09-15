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
    <div className="border-4 border-gray-300 bg-white p-4">
      <h3 className="font-bold text-gray-800 mb-4 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Select Feedback Type</h3>
      <div className="grid grid-cols-2 gap-3">
        {feedbackTypes.map(type => {
          const Icon = type.icon;
          return (
            <button
              key={type.value}
              onClick={() => onTypeChange(type.value)}
              className={`
                p-4 border-4 transition-all duration-200 text-left
                ${selectedType === type.value
                  ? 'border-primary-600 bg-primary-100'
                  : 'border-gray-400 hover:border-gray-500 bg-gray-50'
                }
              `}
            >
              <div className="flex items-center mb-2">
                <Icon className={`h-5 w-5 mr-2 ${selectedType === type.value ? 'text-primary-700' : 'text-gray-600'}`} />
                <span className={`font-bold uppercase text-xs ${selectedType === type.value ? 'text-primary-900' : 'text-gray-800'}`}>
                  {type.label}
                </span>
              </div>
              <p className={`text-sm ${selectedType === type.value ? 'text-primary-800 font-medium' : 'text-gray-600'}`}>
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
        className="w-full px-3 py-2 border-2 border-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-medium"
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
            className="bg-gray-200 shadow-2xl w-[95vw] h-[90vh] max-w-[1600px] overflow-hidden"
          >
            {/* Book Container */}
            <div className="h-full flex flex-col">
              {/* Header */}
              <div className="flex items-center justify-between px-8 py-4 bg-white border-b-4 border-gray-300">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900">
                    Feedback for Record #{record?.id}
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">View details on the left, provide feedback on the right</p>
                </div>

                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-gray-600 transition-colors p-2"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>

              {/* Book Pages Container */}
              <div className="flex-1 flex p-4 gap-2 overflow-hidden bg-gray-200">
                {/* Left Page - Details */}
                <div className="flex-1 bg-white border-4 border-gray-300 overflow-hidden flex flex-col"
                     style={{
                       boxShadow: 'inset 4px 0 12px rgba(0,0,0,0.15), 4px 4px 20px rgba(0,0,0,0.3)',
                       borderRadius: '0 8px 8px 0'
                     }}>
                  {/* Left Page Header */}
                  <div className="px-6 py-4 border-b-4 border-gray-200 bg-gradient-to-r from-gray-50 to-white">
                    <h3 className="text-lg font-bold text-gray-800 uppercase tracking-wide">Record Details</h3>
                  </div>

                  {/* Left Page Content */}
                  <div className="flex-1 p-6 overflow-y-auto bg-gray-50">
                    {/* Record Info */}
                    <div className="space-y-4">
                      <div className="border-4 border-gray-300 bg-white p-4">
                        <h4 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Description</h4>
                        <p className="text-gray-700 leading-relaxed">
                          {record?.description || 'No description available'}
                        </p>
                      </div>

                      {record?.dataset_type && (
                        <div className="border-4 border-gray-300 bg-white p-4">
                          <h4 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Dataset Type</h4>
                          <span className="inline-block px-4 py-2 bg-gray-200 border-2 border-gray-400 text-gray-800 font-bold uppercase text-sm">
                            {record.dataset_type.replace('_', ' ')}
                          </span>
                        </div>
                      )}

                      {/* Additional record details */}
                      {Object.entries(record || {}).filter(([key]) =>
                        !['id', 'description', 'dataset_type'].includes(key)
                      ).map(([key, value]) => (
                        <div key={key} className="border-4 border-gray-300 bg-white p-4">
                          <h4 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">
                            {key.replace(/_/g, ' ')}
                          </h4>
                          <p className="text-gray-700">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : value}
                          </p>
                        </div>
                      ))}

                      {/* Previous Feedback on Left Page */}
                      {existingFeedback.length > 0 && (
                        <div className="border-4 border-gray-300 bg-white p-4">
                          <h4 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Previous Feedback</h4>
                          <div className="space-y-3">
                            {existingFeedback.map((feedback) => (
                              <div key={feedback.id} className="p-3 bg-gray-100 border-2 border-gray-400">
                                <div className="flex items-center justify-between mb-2">
                                  <span className="inline-flex items-center px-3 py-1 text-xs font-bold bg-gray-300 border-2 border-gray-500 text-gray-800 uppercase">
                                    {feedback.feedback_type}
                                  </span>
                                  <span className="text-xs text-gray-600 font-bold">
                                    {new Date(feedback.timestamp).toLocaleDateString()}
                                  </span>
                                </div>
                                <p className="text-gray-800 text-sm font-medium">{feedback.value}</p>
                                {feedback.additional_notes && (
                                  <p className="text-xs text-gray-600 mt-2 italic border-t-2 border-gray-300 pt-2">
                                    Note: {feedback.additional_notes}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Book Spine/Binding */}
                <div className="w-8 bg-gradient-to-r from-gray-400 via-gray-300 to-gray-400 shadow-inner flex items-center justify-center">
                  <div className="h-full w-0.5 bg-gray-600 opacity-30"></div>
                </div>

                {/* Right Page - Feedback Form */}
                <div className="flex-1 bg-white border-4 border-gray-300 overflow-hidden flex flex-col"
                     style={{
                       boxShadow: 'inset -4px 0 12px rgba(0,0,0,0.15), -4px 4px 20px rgba(0,0,0,0.3)',
                       borderRadius: '8px 0 0 8px'
                     }}>
                  {/* Right Page Header */}
                  <div className="px-6 py-4 border-b-4 border-gray-200 bg-gradient-to-l from-gray-50 to-white">
                    <h3 className="text-lg font-bold text-gray-800 uppercase tracking-wide">Provide Feedback</h3>
                  </div>

                  {/* Right Page Content */}
                  <div className="flex-1 p-6 overflow-y-auto bg-gray-50">
                    <form onSubmit={handleSubmit} className="h-full flex flex-col">
                      <div className="flex-1 space-y-4">
                        <FeedbackTypeSelector
                          selectedType={feedbackType}
                          onTypeChange={setFeedbackType}
                          datasetType={record?.dataset_type}
                        />

                        <div className="border-4 border-gray-300 bg-white p-4">
                          <h3 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Your Feedback</h3>
                          {renderFeedbackInput()}
                        </div>

                        {/* Additional Notes */}
                        <div className="border-4 border-gray-300 bg-white p-4">
                          <h3 className="font-bold text-gray-800 mb-3 uppercase text-sm tracking-wide border-b-2 border-gray-200 pb-2">Additional Notes (Optional)</h3>
                          <textarea
                            value={additionalNotes}
                            onChange={(e) => setAdditionalNotes(e.target.value)}
                            placeholder="Any additional context or notes..."
                            className="w-full px-3 py-2 border-2 border-gray-400 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                            rows={3}
                          />
                        </div>
                      </div>

                      {/* Submit Buttons - Fixed at bottom */}
                      <div className="flex justify-end space-x-3 pt-4 mt-auto border-t-4 border-gray-300">
                        <button
                          type="button"
                          onClick={closeModal}
                          className="px-6 py-3 border-4 border-gray-400 bg-gray-200 text-gray-800 font-bold uppercase tracking-wide hover:bg-gray-300 transition-colors"
                          disabled={isSubmitting}
                        >
                          Cancel
                        </button>

                        <button
                          type="submit"
                          disabled={!feedbackValue.trim() || isSubmitting}
                          className="px-6 py-3 border-4 border-primary-600 bg-primary-500 text-white font-bold uppercase tracking-wide hover:bg-primary-600 transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
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
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export { FeedbackTypeSelector, RatingInput, ApprovalInput };
export default FeedbackModal;