import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { apiService } from '../services/api';

const DashboardContext = createContext();

export const useDashboard = () => {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return context;
};

const initialState = {
  selectedDataset: 'external_loss',
  searchQuery: '',
  selectedAiTaxonomy: '',
  selectedErmsTaxonomy: '',
  currentPage: 1,
  pageSize: 50,
  records: [],
  totalRecords: 0,
  totalPages: 0,
  stats: {},
  taxonomies: { ai_taxonomies: [], erms_taxonomies: [] },
  loading: false,
  error: null,
  detailRecord: null,
  feedbackRecord: null,
  combinedViewOpen: false,
};

function dashboardReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    
    case 'SET_SELECTED_DATASET':
      return { 
        ...state, 
        selectedDataset: action.payload,
        currentPage: 1,
        records: [],
        searchQuery: '',
        selectedAiTaxonomy: '',
        selectedErmsTaxonomy: ''
      };
    
    case 'SET_SEARCH_QUERY':
      return { ...state, searchQuery: action.payload, currentPage: 1 };
    
    case 'SET_AI_TAXONOMY':
      return { ...state, selectedAiTaxonomy: action.payload, currentPage: 1 };
    
    case 'SET_ERMS_TAXONOMY':
      return { ...state, selectedErmsTaxonomy: action.payload, currentPage: 1 };
    
    case 'SET_PAGE':
      return { ...state, currentPage: action.payload };
    
    case 'SET_PAGE_SIZE':
      return { ...state, pageSize: action.payload, currentPage: 1 };
    
    case 'SET_RECORDS':
      return {
        ...state,
        records: action.payload.data,
        totalRecords: action.payload.total,
        totalPages: action.payload.total_pages,
        currentPage: action.payload.page,
        loading: false
      };
    
    case 'SET_STATS':
      return { ...state, stats: { ...state.stats, [action.dataset]: action.payload } };
    
    case 'SET_TAXONOMIES':
      return { ...state, taxonomies: action.payload };
    
    case 'SET_DETAIL_RECORD':
      return { ...state, detailRecord: action.payload };
    
    case 'SET_FEEDBACK_RECORD':
      return { ...state, feedbackRecord: action.payload };
    
    case 'SET_COMBINED_VIEW_OPEN':
      return { ...state, combinedViewOpen: action.payload };
    
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    
    default:
      return state;
  }
}

export const DashboardProvider = ({ children }) => {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);

  // Fetch records based on current filters
  const fetchRecords = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const params = {
        page: state.currentPage,
        page_size: state.pageSize,
      };

      if (state.searchQuery) params.search_query = state.searchQuery;
      if (state.selectedAiTaxonomy) params.ai_taxonomy = state.selectedAiTaxonomy;
      if (state.selectedErmsTaxonomy) params.current_erms_taxonomy = state.selectedErmsTaxonomy;

      const response = await apiService.getDatasetRecords(state.selectedDataset, params);
      dispatch({ type: 'SET_RECORDS', payload: response });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  };

  // Fetch dataset statistics
  const fetchStats = async (datasetType) => {
    try {
      const stats = await apiService.getDatasetStats(datasetType);
      dispatch({ type: 'SET_STATS', dataset: datasetType, payload: stats });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  // Fetch taxonomies for filtering
  const fetchTaxonomies = async (datasetType = null) => {
    try {
      const taxonomies = await apiService.getTaxonomies(datasetType);
      dispatch({ type: 'SET_TAXONOMIES', payload: taxonomies });
    } catch (error) {
      console.error('Failed to fetch taxonomies:', error);
    }
  };

  // Fetch record details
  const fetchRecordDetail = async (datasetType, recordId) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const record = await apiService.getRecordDetail(datasetType, recordId);
      dispatch({ type: 'SET_DETAIL_RECORD', payload: record });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  // Actions
  const actions = {
    setSelectedDataset: (dataset) => {
      dispatch({ type: 'SET_SELECTED_DATASET', payload: dataset });
    },
    setSearchQuery: (query) => {
      dispatch({ type: 'SET_SEARCH_QUERY', payload: query });
    },
    setAiTaxonomy: (taxonomy) => {
      dispatch({ type: 'SET_AI_TAXONOMY', payload: taxonomy });
    },
    setErmsTaxonomy: (taxonomy) => {
      dispatch({ type: 'SET_ERMS_TAXONOMY', payload: taxonomy });
    },
    setPage: (page) => {
      dispatch({ type: 'SET_PAGE', payload: page });
    },
    setPageSize: (size) => {
      dispatch({ type: 'SET_PAGE_SIZE', payload: size });
    },
    fetchRecords,
    fetchStats,
    fetchTaxonomies,
    fetchRecordDetail,
    setDetailRecord: (record) => dispatch({ type: 'SET_DETAIL_RECORD', payload: record }),
    setFeedbackRecord: (record) => dispatch({ type: 'SET_FEEDBACK_RECORD', payload: record }),
    setCombinedViewOpen: (isOpen) => dispatch({ type: 'SET_COMBINED_VIEW_OPEN', payload: isOpen }),
    submitFeedback: async (feedback) => {
      try {
        const response = await apiService.submitFeedback(feedback);
        return response;
      } catch (error) {
        dispatch({ type: 'SET_ERROR', payload: error.message });
        throw error;
      }
    },
    clearError: () => dispatch({ type: 'CLEAR_ERROR' })
  };

  // Auto-fetch records when relevant state changes
  useEffect(() => {
    fetchRecords();
  }, [
    state.selectedDataset,
    state.searchQuery,
    state.selectedAiTaxonomy,
    state.selectedErmsTaxonomy,
    state.currentPage,
    state.pageSize
  ]);

  // Fetch taxonomies when dataset changes
  useEffect(() => {
    fetchTaxonomies(state.selectedDataset);
  }, [state.selectedDataset]);

  return (
    <DashboardContext.Provider value={{ state, actions }}>
      {children}
    </DashboardContext.Provider>
  );
};