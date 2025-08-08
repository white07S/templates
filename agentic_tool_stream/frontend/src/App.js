import React from 'react';
import Chat from './Chat';
import './App.css';

function App() {
  const user = 'john.doe';

  return <Chat user={user} />;
}

export default App;
