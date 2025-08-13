export const authService = {
  login: (username, secretCode) => {
    const user = { username, secretCode };
    localStorage.setItem('user', JSON.stringify(user));
    return user;
  },

  logout: () => {
    localStorage.removeItem('user');
    window.location.href = '/login';
  },

  getCurrentUser: () => {
    try {
      return JSON.parse(localStorage.getItem('user') || '{}');
    } catch {
      return {};
    }
  },

  isAuthenticated: () => {
    const user = authService.getCurrentUser();
    return !!(user.username && user.secretCode);
  }
};