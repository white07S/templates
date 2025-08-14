import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Search, Tag, Star } from 'lucide-react';

const PromptLib = () => {
  return (
    <div className="container section">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-4"
      >
        <h1 className="text-title">Prompt Library</h1>
        <p className="text-body">Curated collection of prompts for data analysis and insights</p>
      </motion.div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="card">
          <div className="card-header">
            <BookOpen size={24} />
            <h3 className="text-subheading">Prompt Collection</h3>
          </div>
          <p className="text-body">Coming soon - Extensive library of analysis prompts</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Search size={24} />
            <h3 className="text-subheading">Search & Filter</h3>
          </div>
          <p className="text-body">Coming soon - Advanced search and filtering options</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Tag size={24} />
            <h3 className="text-subheading">Categories</h3>
          </div>
          <p className="text-body">Coming soon - Organized prompt categories</p>
        </div>

        <div className="card">
          <div className="card-header">
            <Star size={24} />
            <h3 className="text-subheading">Favorites</h3>
          </div>
          <p className="text-body">Coming soon - Save and manage your favorite prompts</p>
        </div>
      </div>
    </div>
  );
};

export default PromptLib;