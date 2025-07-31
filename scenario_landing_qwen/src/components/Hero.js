// src/components/Hero.js
import React from 'react';
import { motion } from 'framer-motion';

const Hero = () => {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-[40%] -left-[20%] w-[600px] h-[600px] bg-[#fce8e8] rounded-full blur-3xl opacity-70"></div>
        <div className="absolute -bottom-[40%] -right-[20%] w-[600px] h-[600px] bg-[#f0f0f0] rounded-full blur-3xl opacity-70"></div>
      </div>
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-[#1a1a1a] leading-tight mb-6">
                AI-Powered Risk <span className="text-[#d6001c]">Scenario Intelligence</span>
              </h1>
              <p className="text-xl text-[#4d4d4d] mb-10 max-w-2xl">
                Advanced multi-agent AI system for operational risk management, powered by UBS Quantitative Solutions.
              </p>
              <div className="flex flex-wrap gap-4">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-[#d6001c] hover:bg-[#a80016] text-white font-bold py-3 px-8 rounded-lg transition-all shadow-lg"
                >
                  Generate New Scenario
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="bg-white border border-[#d6001c] text-[#d6001c] hover:bg-[#fce8e8] font-bold py-3 px-8 rounded-lg transition-all"
                >
                  Explore Library
                </motion.button>
              </div>
            </motion.div>
          </div>
          
          <div className="relative">
            <div className="relative aspect-square max-w-md mx-auto">
              <div className="absolute inset-0 bg-gradient-to-br from-[#d6001c] to-[#a80016] rounded-3xl transform rotate-6"></div>
              <div className="absolute inset-0 bg-gradient-to-br from-[#f8f7f5] to-[#f0eeeb] rounded-3xl shadow-2xl p-6 flex flex-col justify-between">
                <div className="flex justify-between">
                  <div className="bg-[#d6001c] w-10 h-10 rounded-full flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div className="text-sm font-medium text-[#d6001c] bg-[#fce8e8] px-3 py-1 rounded-full">
                    AI Generated
                  </div>
                </div>
                
                <div className="mt-6">
                  <h3 className="text-xl font-bold text-[#1a1a1a] mb-3">Market Volatility Scenario</h3>
                  <div className="flex items-center mb-4">
                    <div className="w-8 h-8 bg-[#e6e6e6] rounded-full mr-3"></div>
                    <div>
                      <div className="text-sm font-medium text-[#4d4d4d]">Analysis Agent</div>
                      <div className="text-xs text-[#808080]">Completed 2 min ago</div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-3 mb-6">
                    <div className="bg-[#f0f0f0] p-3 rounded-lg">
                      <div className="text-xs text-[#808080]">Severity</div>
                      <div className="font-bold text-[#d6001c]">High</div>
                    </div>
                    <div className="bg-[#f0f0f0] p-3 rounded-lg">
                      <div className="text-xs text-[#808080]">Probability</div>
                      <div className="font-bold text-[#d6001c]">Medium</div>
                    </div>
                    <div className="bg-[#f0f0f0] p-3 rounded-lg">
                      <div className="text-xs text-[#808080]">Category</div>
                      <div className="font-bold text-[#d6001c]">Market</div>
                    </div>
                  </div>
                </div>
                
                <div className="flex justify-between items-center">
                  <div className="text-sm text-[#4d4d4d]">Scenario ID: SC-2023-0842</div>
                  <button className="text-sm text-[#d6001c] font-medium hover:underline">View Details</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;