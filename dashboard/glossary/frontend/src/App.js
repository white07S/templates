import React from 'react';
import './App.css';
import GlossaryManager from './GlossaryComponents/GlossaryManager';

function App() {
  // You can pass a dynamic user_id here from authentication/context
  const userId = 'user123';

  return (
    <div className="App">
      <GlossaryManager userId={userId} />
    </div>
  );
}

export default App;
