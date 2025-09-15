import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Send } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';
import { apiService } from '../services/api';
import { DetailContent } from './DetailModal';
import { FeedbackTypeSelector, RatingInput } from './FeedbackModal';

const FeedbackForm = ({ record, onClose }) => {
  const { actions } = useDashboard();
  const [feedbackType, setFeedbackType] = useState('text');
  const [feedbackValue, setFeedbackValue] = useState('');
  const [additionalNotes, setAdditionalNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [existingFeedback, setExistingFeedback] = useState([]);

  // Load existing feedback when component mounts
  useEffect(() => {
    if (record) {
      loadExistingFeedback();
    }
  }, [record?.id]);

  const loadExistingFeedback = async () => {
    try {
      const feedback = await apiService.getFeedback(record.id);
      setExistingFeedback(feedback);
    } catch (error) {
      console.error('Failed to load existing feedback:', error);
    }
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
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderFeedbackInput = () => {
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
                px-4 py-2 border transition-all duration-200 capitalize
                ${feedbackValue === option
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-300 hover:border-gray-400 text-gray-700'
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
        className="w-full px-3 py-2 border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
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
    <form onSubmit={handleSubmit} className="h-full flex flex-col">
      <div className="flex-1 space-y-6">
        <FeedbackTypeSelector
          selectedType={feedbackType}
          onTypeChange={setFeedbackType}
          datasetType={record?.dataset_type}
        />

        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Your Feedback</h3>
          <div className="p-4 bg-gray-50 border border-gray-200">
            {renderFeedbackInput()}
          </div>
        </div>

        {/* Additional Notes */}
        <div>
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Additional Notes (Optional)</h3>
          <textarea
            value={additionalNotes}
            onChange={(e) => setAdditionalNotes(e.target.value)}
            placeholder="Any additional context or notes..."
            className="w-full px-3 py-2 border border-gray-300 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            rows={3}
          />
        </div>

        {/* Previous Feedback */}
        {existingFeedback.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold text-gray-900 mb-3">Previous Feedback</h4>
            <div className="space-y-3 max-h-32 overflow-y-auto">
              {existingFeedback.map((feedback) => (
                <div key={feedback.id} className="p-3 bg-gray-50 border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="inline-flex items-center px-2 py-1 text-xs font-medium bg-gray-100 text-gray-800 capitalize">
                      {feedback.feedback_type.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(feedback.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-gray-700 text-sm">{feedback.value}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Submit Buttons - Fixed at bottom */}
      <div className="flex justify-end space-x-3 pt-6 mt-auto border-t border-gray-200">
        <button
          type="button"
          onClick={onClose}
          className="px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
          disabled={isSubmitting}
        >
          Cancel
        </button>

        <button
          type="submit"
          disabled={!feedbackValue.trim() || isSubmitting}
          className="px-4 py-2 bg-primary-600 text-white hover:bg-primary-700 transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
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
  );
};

const CombinedDetailView = () => {
  const { state, actions } = useDashboard();
  const isOpen = !!state.detailRecord;
  const record = state.detailRecord;

  const closeModal = () => {
    actions.setDetailRecord(null);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-start p-4 z-50"
          onClick={closeModal}
        >
          <motion.div
            initial={{ opacity: 0, x: -100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-white shadow-2xl w-[90vw] h-[90vh] overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-200">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  Record #{record?.id}
                </h2>
                <p className="text-sm text-gray-600 mt-1 capitalize">
                  {record?.dataset_type?.replace('_', ' ')} Dataset
                </p>
              </div>

              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 transition-colors p-2"
              >
                <X className="h-6 w-6" />
              </button>
            </div>

            {/* Combined Content */}
            <div className="flex-1 flex overflow-hidden h-[calc(90vh-80px)]">
              {/* Left Half - Details */}
              <div className="flex-1 bg-white border-r border-gray-200 overflow-hidden flex flex-col">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">Record Details</h3>
                </div>
                <div className="flex-1 p-6 overflow-y-auto">
                  <DetailContent record={record} />
                </div>
              </div>

              {/* Right Half - Feedback Form */}
              <div className="flex-1 bg-white overflow-hidden flex flex-col">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">Provide Feedback</h3>
                </div>
                <div className="flex-1 p-6 overflow-y-auto">
                  <FeedbackForm record={record} onClose={closeModal} />
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default CombinedDetailView;