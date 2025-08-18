import React from 'react';
import Ticket from './pages/Ticket/Ticket';
import { motion } from 'framer-motion';

function App() {
  const user = 'testuser';

  return (
    <div className="min-h-screen bg-white">
      <motion.main 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="container mx-auto px-4 py-8"
      >
        <Ticket user={user} />
      </motion.main>
    </div>
  );
}

export default App;