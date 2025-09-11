import React from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Eye, MessageSquare } from 'lucide-react';
import { useDashboard } from '../contexts/DashboardContext';

const TableRow = ({ record, index, onClick, onFeedback }) => {
  return (
    <motion.tr
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.02 }}
      onClick={() => onClick(record)}
      className="hover:bg-gray-50 cursor-pointer transition-colors"
    >
      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
        #{record.id}
      </td>
      <td className="px-6 py-4 text-sm text-gray-900 max-w-md">
        <div className="truncate" title={record.description}>
          {record.description}
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className="inline-flex items-center px-2.5 py-0.5  text-xs font-medium bg-blue-100 text-blue-800">
          {record.ai_taxonomy}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className="inline-flex items-center px-2.5 py-0.5  text-xs font-medium bg-purple-100 text-purple-800">
          {record.current_erms_taxonomy}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
        <div className="flex items-center justify-end space-x-2">
          <button 
            onClick={() => onClick(record)}
            className="text-gray-400 hover:text-primary-600 transition-colors"
            title="View Details"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button 
            onClick={(e) => {
              e.stopPropagation();
              onFeedback(e, record);
            }}
            className="text-gray-400 hover:text-primary-600 transition-colors"
            title="Give Feedback"
          >
            <MessageSquare className="h-4 w-4" />
          </button>
        </div>
      </td>
    </motion.tr>
  );
};

const Pagination = ({ currentPage, totalPages, onPageChange, totalRecords, pageSize }) => {
  const startRecord = (currentPage - 1) * pageSize + 1;
  const endRecord = Math.min(currentPage * pageSize, totalRecords);

  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 5;
    
    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) {
        pages.push(i);
      }
    } else {
      if (currentPage <= 3) {
        pages.push(1, 2, 3, 4, 5);
      } else if (currentPage >= totalPages - 2) {
        for (let i = totalPages - 4; i <= totalPages; i++) {
          pages.push(i);
        }
      } else {
        for (let i = currentPage - 2; i <= currentPage + 2; i++) {
          pages.push(i);
        }
      }
    }
    
    return pages;
  };

  return (
    <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200 sm:px-6">
      <div className="flex-1 flex justify-between sm:hidden">
        <button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1}
          className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages}
          className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>
      
      <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
        <div>
          <p className="text-sm text-gray-700">
            Showing <span className="font-medium">{startRecord}</span> to{' '}
            <span className="font-medium">{endRecord}</span> of{' '}
            <span className="font-medium">{totalRecords}</span> results
          </p>
        </div>
        
        <div>
          <nav className="relative z-0 inline-flex  shadow-sm -space-x-px">
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="relative inline-flex items-center px-2 py-2  border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
            
            {getPageNumbers().map((page) => (
              <button
                key={page}
                onClick={() => onPageChange(page)}
                className={`
                  relative inline-flex items-center px-4 py-2 border text-sm font-medium
                  ${currentPage === page
                    ? 'z-10 bg-primary-50 border-primary-500 text-primary-600'
                    : 'bg-white border-gray-300 text-gray-500 hover:bg-gray-50'
                  }
                `}
              >
                {page}
              </button>
            ))}
            
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="relative inline-flex items-center px-2 py-2  border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          </nav>
        </div>
      </div>
    </div>
  );
};

const DataTable = () => {
  const { state, actions } = useDashboard();

  const handleRecordClick = async (record) => {
    // Fetch full record details and open modal
    await actions.fetchRecordDetail(state.selectedDataset, record.id);
  };

  const handleFeedbackClick = (e, record) => {
    e.stopPropagation();
    actions.setFeedbackRecord(record);
  };

  const handlePageSizeChange = (newSize) => {
    actions.setPageSize(parseInt(newSize));
  };

  if (state.loading && state.records.length === 0) {
    return (
      <div className="animate-pulse">
        <div className="overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left"><div className="h-4 bg-gray-300  w-12"></div></th>
                <th className="px-6 py-3 text-left"><div className="h-4 bg-gray-300  w-32"></div></th>
                <th className="px-6 py-3 text-left"><div className="h-4 bg-gray-300  w-24"></div></th>
                <th className="px-6 py-3 text-left"><div className="h-4 bg-gray-300  w-28"></div></th>
                <th className="px-6 py-3 text-left"><div className="h-4 bg-gray-300  w-16"></div></th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {[...Array(5)].map((_, i) => (
                <tr key={i}>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-200  w-16"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-200  w-80"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-200  w-20"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-200  w-24"></div></td>
                  <td className="px-6 py-4"><div className="h-4 bg-gray-200  w-12"></div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white overflow-hidden">
      {/* Table Controls */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700">
            Show:
            <select
              value={state.pageSize}
              onChange={(e) => handlePageSizeChange(e.target.value)}
              className="ml-2 border border-gray-300  px-3 py-1 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
            </select>
          </label>
        </div>
        
        {state.loading && (
          <div className="flex items-center">
            <div className="animate-spin  h-4 w-4 border-2 border-primary-500 border-t-transparent mr-2"></div>
            <span className="text-sm text-gray-600">Loading...</span>
          </div>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Description
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                AI Taxonomy
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                ERMS Taxonomy
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {state.records.map((record, index) => (
              <TableRow
                key={record.id}
                record={record}
                index={index}
                onClick={handleRecordClick}
                onFeedback={handleFeedbackClick}
              />
            ))}
          </tbody>
        </table>
      </div>

      {/* Empty State */}
      {state.records.length === 0 && !state.loading && (
        <div className="text-center py-12">
          <div className="text-gray-500 text-lg mb-2">No records found</div>
          <div className="text-gray-400 text-sm">
            {state.searchQuery ? 
              `No results matching "${state.searchQuery}"` : 
              'Try adjusting your filters'
            }
          </div>
        </div>
      )}

      {/* Pagination */}
      {state.totalPages > 1 && (
        <Pagination
          currentPage={state.currentPage}
          totalPages={state.totalPages}
          onPageChange={actions.setPage}
          totalRecords={state.totalRecords}
          pageSize={state.pageSize}
        />
      )}
    </div>
  );
};

export default DataTable;