export const API_BASE_URL = 'http://localhost:8000/api';

export const DATA_TYPES = {
  CONTROLS: 'controls',
  ISSUES: 'issues',
  EXTERNAL_LOSS: 'external_loss',
  INTERNAL_LOSS: 'internal_loss',
  ORX_SCENARIOS: 'orx_scenarios'
};

export const TASK_STATUS = {
  PENDING: 'pending',
  COMPLETED: 'completed',
  ERROR: 'error'
};

export const FILE_FORMATS = {
  XLSX: '.xlsx',
  CSV: '.csv'
};

export const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export const REFRESH_INTERVAL = 30000; // 30 seconds

export const ROUTES = {
  HOME: '/',
  TICKET: '/ticket',
  DATA_DASHBOARD: '/data-dashboard',
  CHAT: '/chat',
  PROMPT_LIB: '/prompt-lib',
  LOGIN: '/login'
};