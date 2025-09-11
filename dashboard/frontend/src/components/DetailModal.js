import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, MessageSquare, FileText, Brain, Lightbulb } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';

const JsonViewer = ({ title, data, icon: Icon, color = "blue" }) => {
  if (!data) return null;

  const colorClasses = {
    blue: "bg-blue-50 border-blue-200 text-blue-800",
    green: "bg-green-50 border-green-200 text-green-800", 
    purple: "bg-purple-50 border-purple-200 text-purple-800",
    orange: "bg-orange-50 border-orange-200 text-orange-800"
  };

  return (
    <div className="mb-6">
      <div className="flex items-center mb-3">
        <div className={`p-2 ${colorClasses[color].replace('text-', 'bg-').replace('-800', '-100')}`}>
          <Icon className={`h-5 w-5 ${colorClasses[color].split(' ')[2]}`} />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 ml-3">{title}</h3>
      </div>
      
      <div className={`p-4 border-2 ${colorClasses[color]}`}>
        <pre className="text-sm overflow-x-auto whitespace-pre-wrap">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    </div>
  );
};

const DetailModal = () => {
  const { state, actions } = useDashboard();
  const isOpen = !!state.detailRecord;

  const closeModal = () => {
    actions.setDetailRecord(null);
  };

  const openFeedback = () => {
    actions.setFeedbackRecord(state.detailRecord);
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
                  Record #{state.detailRecord?.id}
                </h2>
                <p className="text-sm text-gray-600 mt-1 capitalize">
                  {state.detailRecord?.dataset_type?.replace('_', ' ')} Dataset
                </p>
              </div>
              
              <div className="flex items-center space-x-3">
                <button
                  onClick={openFeedback}
                  className="btn-primary flex items-center"
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Feedback
                </button>
                
                <button
                  onClick={closeModal}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="h-6 w-6" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto" style={{ maxHeight: 'calc(85vh - 88px)' }}>
              {/* Basic Information */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Description</h3>
                <div className="p-4 bg-gray-50">
                  <p className="text-gray-800 leading-relaxed">{state.detailRecord?.description}</p>
                </div>
              </div>

              {/* Taxonomies */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">AI Taxonomy</h3>
                  <div className="p-4 bg-blue-50 border border-blue-200">
                    <span className="inline-flex items-center px-3 py-1 text-sm font-medium bg-blue-100 text-blue-800">
                      {state.detailRecord?.ai_taxonomy}
                    </span>
                  </div>
                </div>
                
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">ERMS Taxonomy</h3>
                  <div className="p-4 bg-purple-50 border border-purple-200">
                    <span className="inline-flex items-center px-3 py-1 text-sm font-medium bg-purple-100 text-purple-800">
                      {state.detailRecord?.current_erms_taxonomy}
                    </span>
                  </div>
                </div>
              </div>

              {/* JSON Data Sections */}
              <JsonViewer
                title="Raw Metadata"
                data={state.detailRecord?.raw_meta_data}
                icon={FileText}
                color="blue"
              />
              
              <JsonViewer
                title="AI Root Cause Analysis"
                data={state.detailRecord?.ai_root_cause}
                icon={Brain}
                color="green"
              />
              
              <JsonViewer
                title="AI Enrichment"
                data={state.detailRecord?.ai_enrichment}
                icon={Lightbulb}
                color="orange"
              />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default DetailModal;