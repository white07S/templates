import lunr from 'lunr';

// Import all MDX content
const mdxContent = {
  'getting-started': `
Getting Started
Quick start guide to get you up and running
Welcome to our comprehensive documentation. This guide will help you get started with our platform quickly and efficiently.
Prerequisites
Before you begin, ensure you have the following installed:
Node.js version 14 or higher
npm or yarn package manager
Git for version control
Installation
Follow these steps to install and set up the project:
Clone the repository
git clone https://github.com/your-repo/your-project.git
Navigate to the project directory
cd your-project
Install dependencies
npm install
Project Structure
Understanding the project structure helps you navigate the codebase effectively:
project src components pages services utils public docs package.json
Development Workflow
Start Development Environment Ready Install Dependencies Configure Environment Run Development Server Make Changes Test Changes Tests Pass Build Production Deploy Application
Configuration
Configure your environment variables by creating a .env file:
REACT_APP_API_URL=https://api.example.com
REACT_APP_ENV=development
Running the Application
Start the development server:
npm start
The application will be available at http://localhost:3000
Next Steps
Explore the Core Concepts to understand the fundamentals
Check out the API Reference for detailed API documentation
View Examples for practical implementation patterns
  `,
  'core-concepts': `
Core Concepts
Fundamental concepts and architecture
This section covers the fundamental concepts that form the foundation of our platform.
Architecture Overview
Our application follows a modern microservices architecture with the following key components:
Frontend Layer
The presentation layer built with React provides:
Responsive user interface
State management with Context API
Component-based architecture
Real-time data updates
Backend Services
Our backend infrastructure consists of:
RESTful API endpoints
Authentication and authorization
Data processing pipelines
Caching mechanisms
State Management
We use a combination of React hooks and Context API for state management:
StateContext React.createContext
StateProvider children
state setState useState initialState
StateContext.Provider value state setState
children
StateContext.Provider
Data Flow
Understanding how data flows through the application:
User Interaction User performs an action in the UI
State Update Component state is updated
API Call Request sent to backend
Data Processing Backend processes the request
Response Data returned to frontend
UI Update Interface reflects new state
Component Architecture
Our components follow these design principles:
Atomic Design
Atoms Basic building blocks buttons inputs
Molecules Simple component groups
Organisms Complex component sections
Templates Page-level layouts
Pages Complete screens
Component Guidelines
Example component structure
Component props
State and hooks
state setState useState
Effects
useEffect
Side effects
dependencies
Event handlers
handleEvent
Logic
Render
div
JSX
div
Security Considerations
Security is built into every layer:
Input validation and sanitization
HTTPS encryption
JWT token authentication
Role-based access control
API rate limiting
  `,
  'api-reference': `
API Reference
Complete API documentation and endpoints
This section provides comprehensive documentation for all available API endpoints.
Base URL
All API requests should be made to:
https://api.example.com/v1
Authentication
API requests require authentication using Bearer tokens:
headers
Authorization Bearer YOUR_ACCESS_TOKEN
Content-Type application/json
Endpoints
Users
Get User Profile
Retrieve the current user's profile information.
GET /users/profile
Response:
id user_123
name John Doe
email john.doe@example.com
role admin
createdAt 2024-01-01T00:00:00Z
Update User Profile
Update user profile information.
PUT /users/profile
Request Body:
name John Smith
email john.smith@example.com
Data Operations
List Items
Retrieve a paginated list of items.
GET /items?page=1&limit=10&sort=created_desc
Query Parameters:
Parameter Type Description Default
page integer Page number 1
limit integer Items per page 10
sort string Sort order created_desc
filter string Filter criteria -
Response:
data
id item_1
name Item One
description Description of item
status active
pagination
page 1
limit 10
total 100
pages 10
Create Item
Create a new item.
POST /items
Request Body:
name New Item
description Item description
category category_1
Update Item
Update an existing item.
PUT /items/:id
Delete Item
Delete an item.
DELETE /items/:id
Error Handling
API errors follow a consistent format:
error
code VALIDATION_ERROR
message Invalid input data
details
field email
issue Invalid email format
Error Codes
Code HTTP Status Description
UNAUTHORIZED 401 Authentication required
FORBIDDEN 403 Insufficient permissions
NOT_FOUND 404 Resource not found
VALIDATION_ERROR 422 Invalid input data
SERVER_ERROR 500 Internal server error
Rate Limiting
API requests are limited to:
1000 requests per hour for authenticated users
100 requests per hour for unauthenticated users
Rate limit information is included in response headers:
X-RateLimit-Limit 1000
X-RateLimit-Remaining 999
X-RateLimit-Reset 1640995200
  `,
  'examples': `
Examples
Practical examples and code snippets
This section provides practical examples to help you implement common use cases.
React Components
Basic Form Component
A simple form with validation:
ContactForm
formData setFormData useState
name ''
email ''
message ''
errors setErrors useState
handleChange e
name value e.target
setFormData prev
...prev
[name] value
validate
newErrors
formData.name
newErrors.name 'Name is required'
formData.email
newErrors.email 'Email is required'
!/\\S+@\\S+\\.\\S+/.test(formData.email)
newErrors.email 'Email is invalid'
formData.message
newErrors.message 'Message is required'
newErrors
handleSubmit e
e.preventDefault
validationErrors validate
Object.keys validationErrors .length === 0
console.log 'Form submitted:' formData
setErrors validationErrors
form onSubmit handleSubmit
input
type text
name name
value formData.name
onChange handleChange
placeholder Your Name
errors.name && span errors.name span
input
type email
name email
value formData.email
onChange handleChange
placeholder Your Email
errors.email && span errors.email span
textarea
name message
value formData.message
onChange handleChange
placeholder Your Message
errors.message && span errors.message span
button type submit Send Message button
form
Data Fetching with Hooks
Custom hook for API data fetching:
useApi url
data setData useState null
loading setLoading useState true
error setError useState null
useEffect
fetchData async
setLoading true
response await fetch url
response.ok
Error HTTP error status response.status
jsonData await response.json
setData jsonData
err
setError err.message
setLoading false
fetchData
url
data loading error
Usage
UserList
data loading error useApi '/api/users'
loading return div Loading... div
error return div Error: error div
ul
data?.map user
li key user.id user.name li
ul
State Management Patterns
Context with Reducer
Complex state management using Context and useReducer:
Initial state
initialState
user null
theme 'light'
notifications []
Action types
ActionTypes
SET_USER 'SET_USER'
SET_THEME 'SET_THEME'
ADD_NOTIFICATION 'ADD_NOTIFICATION'
REMOVE_NOTIFICATION 'REMOVE_NOTIFICATION'
Reducer
appReducer state action
action.type
ActionTypes.SET_USER
...state user action.payload
ActionTypes.SET_THEME
...state theme action.payload
ActionTypes.ADD_NOTIFICATION
...state
notifications [...state.notifications action.payload]
ActionTypes.REMOVE_NOTIFICATION
...state
notifications state.notifications.filter
n => n.id !== action.payload
state
Context
AppContext createContext
Provider
AppProvider children
state dispatch useReducer appReducer initialState
AppContext.Provider value state dispatch
children
AppContext.Provider
Custom hook
useApp
context useContext AppContext
context
Error 'useApp must be used within AppProvider'
context
Utility Functions
Debounce Function
Prevent excessive API calls:
debounce func delay
timeoutId
function ...args
clearTimeout timeoutId
timeoutId setTimeout
func.apply this args
delay
Usage
debouncedSearch debounce searchTerm
console.log 'Searching for:' searchTerm
Make API call
500
Format Currency
Format numbers as currency:
formatCurrency amount currency 'USD' locale 'en-US'
Intl.NumberFormat locale
style 'currency'
currency currency
.format amount
Usage
console.log formatCurrency 1234.56 $1,234.56
console.log formatCurrency 1234.56 'EUR' 'de-DE' 1.234,56 €
  `,
  'math-formulas': `
Math Formulas Code Examples
Mathematical formulas and enhanced code blocks with copy functionality
This page demonstrates mathematical formula rendering and code blocks with copy functionality.
Mathematical Formulas
Inline Math
You can include inline math formulas like E mc2 or the quadratic formula directly in your text.
Block Math Formulas
For more complex mathematical expressions use block math
Advanced Mathematical Examples
Calculus
The fundamental theorem of calculus
Linear Algebra
Matrix multiplication
Statistics
Normal distribution probability density function
Complex Analysis
Euler formula
Differential Equations
The general solution to a second-order linear homogeneous equation
Greek Letters Symbols
Common mathematical symbols alpha beta gamma Delta Sigma Omega
Summation Integration
Code Blocks Copy Functionality
All code blocks now include a copy button in the top-right corner
JavaScript Python SQL Bash Shell JSON Configuration CSS
Code Block Features
Copy Button Click the copy button in the top-right corner of any code block
Language Detection Automatic syntax highlighting for various programming languages
Language Label Shows the programming language in the top-left corner
Line Numbers Clean formatting with proper indentation
Horizontal Scrolling Long lines can be scrolled horizontally
Math Formula Features
Inline Math Use single dollar signs for inline formulas
Block Math Use triple backticks with math or latex language for block formulas
KaTeX Rendering Fast and accurate mathematical typesetting
Error Handling Graceful fallback for invalid LaTeX syntax
Responsive Math formulas adapt to different screen sizes
Supported Math Elements
Greek letters Operators Relations Functions Calculus Set theory
  `,
  'troubleshooting': `
Troubleshooting
Common issues and their solutions
This guide helps you resolve common issues you might encounter.
Common Issues
Installation Problems
Node Version Mismatch
Problem: Getting errors about incompatible Node.js version.
Solution:
Check your Node version
node --version
Use nvm to install and use correct version
nvm install 14
nvm use 14
Dependency Conflicts
Problem: npm install fails with dependency conflicts.
Solution:
Clear npm cache
npm cache clean --force
Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json
Reinstall dependencies
npm install
Build Errors
Module Not Found
Problem: Build fails with "Module not found" error.
Solution:
1. Check if the module is installed:
   npm list module-name
2. Install missing module:
   npm install module-name
3. Verify import path is correct
Out of Memory
Problem: Build process runs out of memory.
Solution:
Increase Node memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm run build
Runtime Errors
CORS Issues
Problem: API calls blocked by CORS policy.
Solution:
Configure proxy in package.json
proxy https://api.example.com
Or configure CORS headers on server
res.setHeader 'Access-Control-Allow-Origin' '*'
State Update Warnings
Problem: "Can't perform state update on unmounted component" warning.
Solution:
useEffect
isMounted true
fetchData().then data
isMounted
setData data
isMounted false
[]
Performance Issues
Slow Initial Load
Causes and Solutions:
1. Large bundle size
   - Implement code splitting
   - Lazy load components
   - Use dynamic imports
2. Unoptimized images
   - Compress images
   - Use appropriate formats (WebP, AVIF)
   - Implement lazy loading
3. Too many requests
   - Bundle assets
   - Use CDN
   - Enable caching
Memory Leaks
Detection:
Use React DevTools Profiler
Monitor component render times and memory usage
Common Causes:
- Event listeners not removed
- Timers not cleared
- Subscriptions not unsubscribed
Prevention:
useEffect
timer setInterval
Do something
1000
Cleanup
clearInterval timer
[]
Debugging Tips
Browser DevTools
1. Console Debugging
   console.log 'Data:' data
   console.table arrayData
   console.trace 'Trace call stack'
2. Breakpoints
   - Add debugger; statement in code
   - Set breakpoints in Sources tab
3. Network Tab
   - Monitor API calls
   - Check response data
   - Verify headers
React Developer Tools
- Inspect component tree
- View props and state
- Profile performance
- Track renders
Getting Help
If you're still experiencing issues:
1. Check the GitHub Issues
2. Search Stack Overflow
3. Join our Discord Community
4. Contact support at support@example.com
Reporting Bugs
When reporting issues, please include:
1. Error message and stack trace
2. Steps to reproduce
3. Environment details (OS, Node version, browser)
4. Relevant code snippets
5. Expected vs actual behavior
  `
};

// Create search index
export const createSearchIndex = () => {
  const documents = Object.entries(mdxContent).map(([id, content]) => {
    // Extract title from content (first line)
    const lines = content.trim().split('\n');
    const title = lines.find(line => line.trim().length > 0)?.trim() || id;

    // Clean content: remove extra whitespace and normalize
    const cleanContent = content
      .replace(/\n+/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    return {
      id,
      title,
      content: cleanContent,
      path: `/docs/${id}`,
      // Extract description (usually second non-empty line)
      description: lines[1]?.trim() || ''
    };
  });

  const index = lunr(function() {
    this.ref('id');
    this.field('title', { boost: 5 });
    this.field('description', { boost: 3 });
    this.field('content', { boost: 1 });

    documents.forEach(doc => {
      this.add(doc);
    });
  });

  return { index, documents };
};

// Search function
export const searchDocuments = (query, { index, documents }) => {
  if (!query || query.length < 2) return [];

  try {
    const results = index.search(`${query}*`);

    return results.map(result => {
      const doc = documents.find(d => d.id === result.ref);
      return {
        ...result,
        document: doc
      };
    }).slice(0, 5); // Limit to top 5 results
  } catch (error) {
    console.error('Search error:', error);
    return [];
  }
};