# Glossary Management System

A web application for managing organizational glossary terms with review workflow.

## Features

### Backend
- **FastAPI** REST API with automatic documentation
- **TinyDB** for fast, file-based data storage
- Full CRUD operations for glossary terms
- Duplicate term validation
- Search functionality with partial matching
- Temp database for change review workflow
- Automatic timestamp tracking (createdAt, updatedAt)

### Frontend
- **React** application with **Tailwind CSS** styling
- Clean, bank-style UI (no rounded corners, black/white/red color scheme)
- Real-time search functionality
- Create, Read, Update, Delete operations
- Form validation
- Responsive design

## Installation

### Prerequisites
- Python 3.9+
- Node.js 14+
- npm or yarn

### Backend Setup
```bash
cd backend
python3 -m pip install -e .
```

### Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### Start Backend Server
```bash
cd backend
python3 -m uvicorn app:app --reload --port 8000
```
The backend will be available at http://localhost:8000
API documentation available at http://localhost:8000/docs

### Start Frontend Server
```bash
cd frontend
npm start
```
The frontend will be available at http://localhost:3000

## API Endpoints

- `GET /api/glossary/terms` - Get all terms
- `GET /api/glossary/terms/{id}` - Get single term by ID
- `POST /api/glossary/terms` - Create new term
- `PUT /api/glossary/terms/{id}` - Update term
- `DELETE /api/glossary/terms/{id}` - Delete term
- `GET /api/glossary/search?q={query}` - Search terms
- `GET /api/glossary/pending-changes` - View pending changes

## Usage

1. **View Terms**: The main page displays all glossary terms in a table
2. **Search**: Use the search bar to find terms (partial matching supported)
3. **Create Term**: Click "Add New Term" and fill in the form
4. **Edit Term**: Click "Edit" on any term to modify it
5. **View Details**: Click "View" to see full term details
6. **Delete Term**: Click "Delete" to remove a term (with confirmation)

## Data Storage

- Main database: `backend/data/glossary.json`
- Temp database (for review): `backend/data/temp_glossary.json`

## User ID

Currently uses a static user ID (`user123`). In production, this should be integrated with an authentication system.

## Development Notes

- All changes are tracked in both main and temp databases
- The temp database maintains an audit trail for manual review
- Duplicate terms are prevented at the API level
- Terms are sorted alphabetically in the list view