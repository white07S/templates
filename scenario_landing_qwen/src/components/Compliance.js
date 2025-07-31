// src/components/Compliance.js
import React from 'react';
import { motion } from 'framer-motion';

const Compliance = () => {
  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-[#1a1a1a] mb-4">Security & Compliance</h2>
          <p className="text-lg text-[#4d4d4d] max-w-3xl mx-auto">
            Enterprise-grade security measures ensuring regulatory compliance and data protection
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="bg-gradient-to-br from-[#f8f7f5] to-[#f0eeeb] rounded-2xl p-8 border border-[#e6e6e6]"
          >
            <div className="flex items-center mb-6">
              <div className="bg-[#d6001c] p-3 rounded-lg mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-[#1a1a1a]">Data Sovereignty</h3>
            </div>
            <p className="text-[#4d4d4d] mb-6">
              All AI processing utilizes local LLMs within UBS infrastructure, ensuring complete data control and compliance with financial regulations.
            </p>
            <div className="flex items-center">
              <div className="mr-4">
                <div className="w-12 h-12 bg-[#e6e6e6] rounded-full flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-[#d6001c]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
              </div>
              <div>
                <div className="font-medium text-[#1a1a1a]">Certified Infrastructure</div>
                <div className="text-sm text-[#808080]">ISO 27001, SOC 2 Type II, GDPR compliant</div>
              </div>
            </div>
          </motion.div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm"
            >
              <div className="flex items-center mb-4">
                <div className="bg-[#fce8e8] p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016zM12 9v2m0 4h.01" />
                  </svg>
                </div>
                <h4 className="font-bold text-[#1a1a1a]">Data Classification</h4>
              </div>
              <p className="text-[#4d4d4d] text-sm">
                No Confidential Information Data (CID) is processed by AI systems. All inputs are classified as Public or Internal.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm"
            >
              <div className="flex items-center mb-4">
                <div className="bg-[#fce8e8] p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <h4 className="font-bold text-[#1a1a1a]">User Responsibility</h4>
              </div>
              <p className="text-[#4d4d4d] text-sm">
                Users maintain responsibility for validating AI outputs and ensuring compliance with all regulatory requirements.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm"
            >
              <div className="flex items-center mb-4">
                <div className="bg-[#fce8e8] p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h4 className="font-bold text-[#1a1a1a]">Audit Trails</h4>
              </div>
              <p className="text-[#4d4d4d] text-sm">
                Comprehensive audit logs for all scenario generation activities with immutable records for compliance verification.
              </p>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="bg-white border border-[#e6e6e6] rounded-2xl p-6 shadow-sm"
            >
              <div className="flex items-center mb-4">
                <div className="bg-[#fce8e8] p-2 rounded-lg mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <h4 className="font-bold text-[#1a1a1a]">Access Control</h4>
              </div>
              <p className="text-[#4d4d4d] text-sm">
                Role-based access controls with multi-factor authentication and granular permission management.
              </p>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Compliance;