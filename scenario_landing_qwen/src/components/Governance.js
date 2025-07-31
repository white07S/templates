// src/components/Governance.js
import React from 'react';
import { motion } from 'framer-motion';

const team = [
  { 
    name: "Tanja Weiher", 
    role: "Model Sponsor", 
    initials: "TW",
    department: "Risk Management",
    email: "tanja.weiher@ubs.com"
  },
  { 
    name: "Anais Dangel", 
    role: "Model Owner", 
    initials: "AD",
    department: "Quantitative Solutions",
    email: "anais.dangel@ubs.com"
  },
  { 
    name: "Eric Cope", 
    role: "Developer", 
    initials: "EC",
    department: "AI Engineering",
    email: "eric.cope@ubs.com"
  },
  { 
    name: "Pablo Mera", 
    role: "Developer", 
    initials: "PM",
    department: "AI Engineering",
    email: "pablo.mera@ubs.com"
  },
  { 
    name: "Preetam Sharma", 
    role: "Lead Developer", 
    initials: "PS",
    department: "AI Engineering",
    email: "preetam.sharma@ubs.com"
  }
];

const Governance = () => {
  return (
    <section className="py-20 bg-gradient-to-b from-[#f8f7f5] to-[#f0eeeb]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-[#1a1a1a] mb-4">Model Governance</h2>
          <p className="text-lg text-[#4d4d4d] max-w-3xl mx-auto">
            Experienced team ensuring robust risk management and regulatory compliance
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-16">
          {team.map((member, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              className="bg-white rounded-2xl shadow-lg overflow-hidden"
            >
              <div 
                className="h-24 w-full"
                style={{ backgroundColor: '#d6001c' }}
              ></div>
              <div className="p-6 -mt-12">
                <div className="flex justify-center">
                  <div className="bg-white border-4 border-white rounded-full p-1 shadow-lg">
                    <div className="bg-[#fce8e8] w-16 h-16 rounded-full flex items-center justify-center">
                      <span className="text-xl font-bold text-[#d6001c]">{member.initials}</span>
                    </div>
                  </div>
                </div>
                <div className="text-center mt-6">
                  <h3 className="text-xl font-bold text-[#1a1a1a]">{member.name}</h3>
                  <p className="text-[#d6001c] font-medium">{member.role}</p>
                  <p className="text-sm text-[#4d4d4d] mt-2">{member.department}</p>
                  <a 
                    href={`mailto:${member.email}`} 
                    className="text-sm text-[#808080] hover:text-[#d6001c] block mt-3"
                  >
                    {member.email}
                  </a>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
        
        <div className="bg-white rounded-2xl shadow-lg p-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-2xl font-bold text-[#1a1a1a] mb-6">Governance Framework</h3>
              <div className="space-y-4">
                <div className="flex">
                  <div className="mr-4">
                    <div className="w-10 h-10 bg-[#fce8e8] rounded-full flex items-center justify-center text-[#d6001c]">
                      1
                    </div>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#1a1a1a]">Model Development</h4>
                    <p className="text-sm text-[#4d4d4d]">Strict protocols for AI model training and validation</p>
                  </div>
                </div>
                
                <div className="flex">
                  <div className="mr-4">
                    <div className="w-10 h-10 bg-[#fce8e8] rounded-full flex items-center justify-center text-[#d6001c]">
                      2
                    </div>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#1a1a1a]">Implementation Review</h4>
                    <p className="text-sm text-[#4d4d4d]">Quarterly reviews by independent validation team</p>
                  </div>
                </div>
                
                <div className="flex">
                  <div className="mr-4">
                    <div className="w-10 h-10 bg-[#fce8e8] rounded-full flex items-center justify-center text-[#d6001c]">
                      3
                    </div>
                  </div>
                  <div>
                    <h4 className="font-bold text-[#1a1a1a]">Performance Monitoring</h4>
                    <p className="text-sm text-[#4d4d4d]">Continuous monitoring of model performance and drift</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-[#f8f7f5] to-[#f0eeeb] rounded-xl p-6">
              <h4 className="font-bold text-[#1a1a1a] mb-4">Documentation & Compliance</h4>
              <ul className="space-y-2 text-sm text-[#4d4d4d]">
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c] mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span>SR 11-7 / OCC 2021-39 compliance</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c] mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span>Full model documentation available</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c] mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span>Annual model validation reports</span>
                </li>
                <li className="flex items-start">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#d6001c] mr-2 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span>Regular regulatory engagement</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Governance;